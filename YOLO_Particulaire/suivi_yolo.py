from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt


def detection_personne(img, net, output_layers):
    height, width, channels = img.shape

    """Faire la detection """

    # Extraction des blobs de l'image 
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # Faire la détection en donnant à Yolo comme entré les blobs 
    net.setInput(blob)
    outs = net.forward(output_layers)

    """Récupérer les coordonnée des boxes des région detectée ansi que leur classe et la précision de la detection """ 
    
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            # Précision de la détection de chaque classe
            scores = detection[5:]
            # Récupérer l'ID de la classe détectée
            class_id = np.argmax(scores)
            # Précision de la détection 
            confidence = scores[class_id]
            # Vérifier que la classe détecter est bien une personne et que la précision de la détection est 
            # supérieur à 0.5
            if (confidence > 0.8) and (class_id == 0):
                # Récupérer le centre et la largeur et la longueur de l'objet détecté 
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Récupérer les coordonnées de point de début de rectangle qui englobe l'objet détecté 
                x = int(center_x - int(w / 2))
                if x < 0 :
                    x = 0
                y = int(center_y - int(h / 2))
                if y < 0 :
                    y=0
                # Append les cordonnées de l'objet ainsi que la précision de la détection et l'ID de la classe détecté
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes,img

# Fonction pour calculer l'histogramme de la composante teinte des régions d'intérêt détecter 
def hsv_histogram_for_window(frame, window):
    # Récupérer la position de cadre qui contient la personne.
    c,r,w,h = window
    # Récupérer la partie de frame ou la personne est détecté. 
    personne = frame[r:r+h,c:c+w]
    # Convertir la partie de frame récupérer en espace couleur HSV.
    personne_hsv =  cv2.cvtColor(personne, cv2.COLOR_RGB2HSV)
    # Application un seuillage sur la partie récupérer de frame. 
    personne_threshold = cv2.inRange(personne_hsv, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    # Calcule de l'histogramme de la composante teinte   
    personne_hist = cv2.calcHist([personne_hsv],[0],personne_threshold,[180],[0,180])
    # Normalisation de l'histogramme 
    cv2.normalize(personne_hist,personne_hist,0,255,cv2.NORM_MINMAX)
    return personne_hist

# Fonction de rééchantillonnage 
def resample(weights):
    n = len(weights)
    indices = []
    # Créer un tableau ou C[i]= sum(w[0:i])
    C = [0.] + [sum(weights[:i+1]) for i in range(1,n)]
    # Déclarer une variable aléatoire 
    u0 = np.random.uniform(0,1/n)
    j = 0
    """ Création d'un tableau ou chaque case  « i = u0+(i-1)/n » et pour chaque u de ce tableau on 
    compare s'il est supérieur que la case «j» de la table «C» crée déjà («j» est un entier initialisé 
    à 0) si c'est le cas nous incrémentons «j» jusqu'à ou deviens < C[j] et là on ajoute l'indice 
    «j-1» à la table des indices (qui présente les indices des particule à garder) """
    for u in [u0+(i-1)/n for i in range(n)]:
        while u > C[j]:
            j+=1
        indices.append(j-1)
    return indices

# Fonction qui extrait les valeurs de l'image BackProjection des coordonnées des particule
def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]

# Fonction pour le calcul de la distance de Matusita entre deux histogrammes
def distance_Matusita(histogram_image1,histogram_image2): 
    histogram_image1 = np.asarray(histogram_image1)
    histogram_image2 = np.asarray(histogram_image2)
    # Calcule la distance entre les deux images en utilisant la formule de la distance Matusita
    dist = pow (sum (pow((np.sqrt(histogram_image1)-np.sqrt(histogram_image2)),2)),0.5)
    return dist

# Fonction qui performe le suivi
def particle_filtre_traking(video):
    """
    Initialisation de suivi 
    """

    """ Performer la détection des personnes """
    # Charger Yolo
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    # Initialisé les classes depuis un fichier 
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    # Extraction des layer 
    layer_names = net.getLayerNames()
    # Extraction des layer des sorties
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # Crée la couleur
    color = np.random.uniform(0, 255, size=(1, 3))
    font = cv2.FONT_HERSHEY_PLAIN
    # Lecture des frames de vidéo 
    ret, image = video.read()
    # Initialiser de nombre de frame 
    frameCounter = 1
    # Détection des personnes dans le premier frame 
    pick, frame = detection_personne(image, net, output_layers)
    while pick == []:
        ret, frame = video.read()
        pick, frame = detection_personne(frame, net, output_layers)

    """Élimination des boxes redondants qui représente la même région et Association des ID au box"""
      
    box=[]
    personnes_id_box= []
    ID = 0
    #Supprimer les boxes doublant et association de ID pour le premier frame
    personnes_id_box.append([ID, pick[0]])
    box.append(pick[0])
    for i in range(len(pick)-1):
        existe = False
        for j in range(len(box)):
            if abs(pick[i+1][0] - box[j][0]) < 20 and abs(pick[i+1][1] - box[j][1]) < 20:
                existe = True
        if not(existe):
            box.append(pick[i+1])
            ID+=1
            personnes_id_box.append([ID,pick[i+1]])
    
    """ Initialisation de l'algorithme de Traking filtre particule """

    # Initialiser le nombre de particule à crée 
    n_particles = 300
    # Calcule des histogrammes des région qui contient des personnes 
    roi_hist = []
    for (c,r,w,h) in box:
        roi_hist.append(hsv_histogram_for_window(frame, (c,r,w,h)) )
    # Calcule de la position initiale
    init_pos = []
    for (c,r,w,h) in box:
        init_pos.append(np.array([c + int(w/2.0),r +int(h/2.0) ], int) )
    # Initialiser les particules initiales pour les positions initiales
    particles = []
    for i in range(len(init_pos)):
        particles.append(np.ones((n_particles, 2), int) * init_pos[i]) 
    # Initialiser les poids des particules d'une façon qu'il soit tous uniforme au début 
    weights = []
    for j in range(len(particles)):
        weights.append(np.ones(n_particles) / n_particles )
    # Initialisation de pas à utiliser pour calculer la distribution des particules 
    stepsize = 15



    """
    * Parcourir la vidéo frame par frame
    * Faire le suivi avec la réinitialisation de la détection après un certain nombre de frame 
    """

    while(1):
        """
        lecture de frame 

        """
        # lire un autre frame 
        ret ,frame = video.read() 
        if ret == False:
            break
        """
        Réinitialisation de la détection après un certain nombre de frame 

        """
        if frameCounter % 100 == 0 :


            """ Détection de la personne """
            # Détection des personnes dans un frame  
            pick = []
            pick, frame = detection_personne(frame, net, output_layers)
            while pick == []:
                ret, frame = video.read()
                pick, frame = detection_personne(frame, net, output_layers)


            """
             Élimination des boxes redondants qui représente la même région et Association des ID au box
             
             """
            # Initialiser 2 listes 
            new_personnes_id_box= [] # Liste des nouvelles box avec des nouveau ID
            box=[] # Liste des boxes

            
            # Calcule l'histogramme de la première zone ou une personne est détecté
            hist_nv_personne = hsv_histogram_for_window(frame, (pick[0][0],pick[0][1],pick[0][2],pick[0][3]))
            # Calcule de la distance entre cet histogramme et les histogrammes des zones détecté dans la détection précédante 
            distance = []  
            new = True
            for k in range(len(roi_hist)):
                distance.append(distance_Matusita(roi_hist[k],hist_nv_personne)) 
            
            # Calcule la correspondance entre la nouvelle zone et les zones détectées dans la détection précédente
            correspond = distance/np.sum(distance) * len(distance)
            
            # Parcourir la table de correspondance et si une des correspondances est inférieur à 0.9 
            # Alors le box prend le même ID que le box qui le correspond dans la détection d'avant 
            # Sinon on lui associe un nouveau ID 
           
            for k in range(len(correspond)):
                if correspond[k][0] < 0.9 :
                    new_personnes_id_box.append([personnes_id_box[k][0],pick[0]])
                    new = False
                    break
            if new :
                ID += 1
                new_personnes_id_box.append([ID,pick[0]])
            # Ajouter le premier Box détecter à la liste des boxes 
            box.append(pick[0])
            """ Faire l'élimination des boxes doublant, ainsi que l'association des ID pour les nouveaux box détecter"""
            for i in range(len(pick)-1):
                # vérifier si les cordonnée de nouveau box détecter est trop proche de celle des box déjà détecter
                # si c'est le cas alors le box est marker comme existe et ne sera pas ajouter à la liste des box
                # sinon le box est ajouté et on lui associer un ID
                existe=False
                for j in range(len(box)):
                    if abs(pick[i+1][0] - box[j][0])/box[j][2] < 0.2 and abs(pick[i+1][1] - box[j][1])/box[j][3] < 0.2:
                        existe = True
                if not(existe):
                    # Ajouter le box à la liste des boxes
                    box.append(pick[i+1])
                    # Calculé l'histogramme de la région inclut dans le box
                    hist_nv_personne = hsv_histogram_for_window(frame, (pick[i+1][0],pick[i+1][1],pick[i+1][2],pick[i+1][3]))
                    #Calculer la distance entre l'histogramme de cette région et les histogrammes calculé dans la detection précédante
                    new = True
                    distance = []
                    for k in range(len(roi_hist)):
                        distance.append(distance_Matusita(roi_hist[k],hist_nv_personne))
                    # calculé la correspandance entre l'histogramme de cette région et les histogrammes calculé dans la detection précédante
                    correspond = distance/np.sum(distance) * len(distance)
                    # Parcourir la table de correspondance et si une des correspondances est inférieur à 0.9 
                    # Alors le box prend le même ID que le box qui le correspond dans la détection d'avant 
                    # Sinon on lui associe un nouveau ID
                    for k in range(len(correspond)):
                        if correspond[k][0] < 0.9 :
                            new_personnes_id_box.append([personnes_id_box[k][0],pick[i+1]])
                            new = False
                            break
                    if new :
                        ID += 1
                        new_personnes_id_box.append([ID,pick[i+1]])
            # Mis à jour la liste des personnes avec les IDs 
            personnes_id_box = new_personnes_id_box           
            
            """Calcule de l'histogramme des zones d'intérêt"""
            # Calcule des histogrammes des patch qui contient des personnes 
            roi_hist = []
            for (c,r,w,h) in box:
                roi_hist.append(hsv_histogram_for_window(frame, (c,r,w,h)))
            
            """Initialisation des particule leur position et leur poids """
            # Calcule de la position initiale
            init_pos = []
            for (c,r,w,h) in box:
                init_pos.append(np.array([c + w/2.0,r + h/2.0], int) )
            # Initialiser les particules initiales pour les positions initiales 
            particles = []
            for i in range(len(init_pos)):
                particles.append(np.ones((n_particles, 2), int) * init_pos[i]) 
            # Initialiser les poids des particules d'une façon qu'il soit tous uniforme au début 
            weights = []
            for j in range(len(particles)):
                weights.append(np.ones(n_particles) / n_particles )


        """
            Faire le suivi 
        """

        """ Diffusion des particules """
        # Diffuser les particules en ajoutant des échantillons d'une distribution uniforme .
        for j in range(len(particles)):
            np.add(particles[:][j], np.random.uniform(-stepsize, stepsize, particles[:][j].shape), out=particles[:][j], casting="unsafe")
         # Limité les coordonnées des particules créent à la taille de frame pour ne pas avoir un débordement 
        for j in range(len(particles)):
            particles[j] = particles[j].clip(np.zeros(2), np.array((frame.shape[1],frame.shape[0]))-1).astype(int) 
        
        """ Calcule des poids de chaque particule """
        # Convertir le frame à l'espace couleur HSV
        Frame_HSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        # Calcule l'image de propagation arrière   
        Image_BP = []
        for j in range(len(roi_hist)):
            Image_BP.append(cv2.calcBackProject([Frame_HSV],[0],roi_hist[j],[0,180],1))
       # Calcule la correspondance entre la position des particules et la position de la région d'intérêt initialisé 
        f = []
        for j in range(len(Image_BP)):
            f.append(particleevaluator(Image_BP[j], particles[j].T))
        # Le calcul des poids de chaque particule selon sa correspondance avec la position de la région d’intérêt initialisée
        for j in range(len(f)):
            weights[j] = np.float32(f[j].clip(1)) 
            weights[j] /= np.sum(weights[j])
        
        
        """ Vérifier la dégénération et rééchantillonner  """
        # Vérifier si y'as une dégénération importante des particules
        Nt = 100
        for j in range(len(particles)) :
            
            if 1. / np.sum(weights[j]**2) < Nt :
                # Rééchantillonner les indices pour avoir les indices des particules qui ont un grand poids    
                particles[j] = particles[j][resample(weights[j]),:]
        
        """Calcul de l'état moyenne et le prendre pour une région d'intérêt """
        # Calcule de l'état moyenne
        pos = []
        for j in range(len(particles)) :
            pos.append(np.sum(particles[:][j].T * weights[j], axis=1).astype(int))
        
        """ Tracer les rectangles sur les régions d'intérêt """
        # Dessiner le rectangle qui englobe la personne 
        for j in range(len(pos)):
            label = str(personnes_id_box[j][0])
            cv2.rectangle(frame,(int(pos[j][0]-personnes_id_box[j][1][2]/2.0) ,int(pos[j][1]-personnes_id_box[j][1][3]/2)),(int(pos[j][0]+personnes_id_box[j][1][2]/2),int(pos[j][1]+personnes_id_box[j][1][3]/2)),(0,255,0),3)
            cv2.putText(frame, label, (int(pos[j][0]-box[j][2]/2.0), int(pos[j][1]-box[j][3]/2) + 30), font, 3, color[0], 3)
        
                
        cv2.imshow('traking_people',frame)
        k = cv2.waitKey(25) & 0xff
         
        frameCounter = frameCounter + 1



        

""" 
Lecture de la vidéo et application de suivi 
"""

video = cv2.VideoCapture('people1.mp4')
particle_filtre_traking(video)  

video.release()
cv2.destroyAllWindows()