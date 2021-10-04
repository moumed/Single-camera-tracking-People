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


    # extraction des blob de l'image 
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # faire la detection on donnant à yolo comme entré les blob 
    net.setInput(blob)
    outs = net.forward(output_layers)
    # dessiner les résultat sur l'image 
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            # précision de la détection de chaque classe
            scores = detection[5:]
            # récupérer l'id de la classe détectée
            class_id = np.argmax(scores)
            # précision de la détection 
            confidence = scores[class_id]
            # vérifier que la classe detecter est bien une personne et que la précision de la detection est 
            # supérieur à 0.5
            if (confidence > 0.5) and (class_id == 0):
                # récupérer le centre et la largeur et la longueur de l'objet detecter 
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                print(center_x, center_y, w , h)
                # récupérer les coordonnée de point de début de rectangle qui englobe l'objet detecter 
                x = int(center_x - int(w / 2)+4)
                y = int(center_y - int(h / 2)+4)
                # append les cordonnée de l'objet ainsi que la précision de la detection et l'id de la classe detecter 
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    print(boxes)
    return boxes,img

def hsv_histogram_for_window(frame, window):
    # initialiser la position de cadre qui contien le visage
    c,r,w,h = window
    # récupérer la partie de visage depuis le frame de la vidéo 
    visage = frame[r:r+h, c:c+w]
    print(r,h)
    # convertire l'image de la partie de visage en espace couleur HSV
    visage_hsv =  cv2.cvtColor(visage, cv2.COLOR_RGB2HSV)
    # application un seuillage sur la partie visage de frame 
    visage_threshold = cv2.inRange(visage_hsv, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    # calcule de l'histograme 
    visage_hist = cv2.calcHist([visage_hsv],[0],visage_threshold,[180],[0,180])
    # normalisation de l'histogramme 
    cv2.normalize(visage_hist,visage_hist,0,255,cv2.NORM_MINMAX)
    return visage_hist


def resample(weights):
    n = len(weights)
    indices = []
    # créer un tableau ou C[i]= sum(w[0:i])
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    # déclarer une variable aléatoire 
    u0 = np.random.random()
    j = 0
    """ création d'un tableau ou chaque case i = (u0+i)/n 
        et pour chaque u de ce tableau on compare s'il est supérieur que la case j de la table C crée déjà (j est un entier
        initialisé à 0) si c'est le cas on incrémente j jusqu'à ou deviens < C[j] et là on ajoute l'indice j-1 à la table des
        indice (qui présente les indices des particule à garder) """
    for u in [(u0+i)/n for i in range(n)]:
        while u > C[j]:
            j+=1
        indices.append(j-1)
    return indices


def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]


# Particle Filter
def particle_filtre_traking(video):
    #initialisation 
    roi_hist = []
    weights = [[]]
    particles = []
    init_pos = []
    n_particles = []
    # Inatialiser le clasifieur SVM à base de detecteur HoG
    # Charger Yolo
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    # initialisé les classe depuis un fichier 
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # extraction des layer 
    layer_names = net.getLayerNames()

    # extraction des layer des sorties
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # crée les couleur pour chacune des classe 
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    

    # initialiser de nombre de frame 
    frameCounter = 0
    # lecture des frame de vidéo 
    ret, image = video.read()
    # detection des personne dans le premier frame 
    pick, frame = detection_personne(image, net, output_layers)
    print(pick)
    while pick == []:
        ret, image = video.read()
        pick, frame = detection_personne(image, net, output_layers)
    # initialiser le nombre de particule à crée 
    n_particles = 50
    # calcule de les histogram des patch qui contient des personnes 
    for (c,r,w,h) in pick:
        roi_hist.append(hsv_histogram_for_window(frame, (c,r,w,h)) )
    # calcule de la position initial
    for (c,r,w,h) in pick:
        init_pos.append(np.array([c + w/2.0,r + h/2.0], int) )
    # initialiser les particule initial pour les position initial
    for i in range(len(init_pos)):
        particles.append(np.ones((n_particles, 2), int) * init_pos[i]) 
    # initialiser les poids des partcule d'une façon qu'il soit tous uniforme au début 
    for j in range(len(particles)):
        weights.append(np.ones(n_particles) / n_particles )
    # initialisation de pas à utilisé pour calculé la distribution des particule 
    stepsize = 15
 
    while(1):
        ret ,frame = video.read()
        if frameCounter % 50 == 0 :
            ret ,image = video.read()

            # lire le frame de la vidéo 
            pick = []
            pick, frame = detection_personne(image, net, output_layers)
            # calcule de les histogram des patch qui contient des personnes 
            roi_hist = []
            for (c,r,w,h) in pick:
                roi_hist.append(hsv_histogram_for_window(frame, (c,r,w,h)))
            # calcule de la position initial
            init_pos = []
            for (c,r,w,h) in pick:
                init_pos.append(np.array([c + w/2.0,r + h/2.0], int) )
            # initialiser les particule initial pour les position initial 
            particles = []
            for i in range(len(init_pos)):
                particles.append(np.ones((n_particles, 2), int) * init_pos[i]) 
            # initialiser les poids des partcule d'une façon qu'il soit tous uniforme au début 
            weights = []
            for j in range(len(particles)):
                weights.append(np.ones(n_particles) / n_particles )


        # lire un autre frame 
        ret ,frame = video.read() 
        if ret == False:
            break
        # diffuser les particules en ajoutant des échantillons d'une distribution uniforme. 
        for j in range(len(particles)):
            np.add(particles[j], np.random.uniform(-stepsize, stepsize, particles[j].shape), out=particles[j], casting="unsafe")
        # convertir le frame à l'espace couleur HSV
        hsvt = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        # Calcule l'image de propagation arriére 
        hist_bp = []
        for j in range(len(roi_hist)):
            hist_bp.append(cv2.calcBackProject([hsvt],[0],roi_hist[j],[0,180],1)) 
        # limité les coordonnée des particule crée à la taille de frame pour ne pas avoir un débordement 
        for j in range(len(particles)):
            particles[j] = particles[j].clip(np.zeros(2), np.array((frame.shape[1],frame.shape[0]))-1).astype(int) 
        # calcule la correspandance entre la position des particul et la position de visage 
        f = []
        for j in range(len(hist_bp)):
            f.append(particleevaluator(hist_bp[j], particles[j].T))
        # le calcule des poids de chaque particule selon sa correspandance avec la position de visage
        for j in range(len(f)):
            weights[j] = np.float32(f[j].clip(1)) 
            weights[j] /= np.sum(weights[j])
        # verifier si y'as une dégénération importante des particule
        for j in range(len(particles)) :
            if 1. / np.sum(weights[j]**2) < n_particles :
                # Rechantilloner les indices pour avoir les indices des particule qui ont un grand poinds   
                particles[j] = particles[j][resample(weights[j]),:]
        # calcule de l'etat moyenne
        pos = []
        for j in range(len(particles)) :
            pos.append(np.sum(particles[j].T * weights[j], axis=1).astype(int))
        #dessiner le rectangle qui englobe le visage 
        for j in range(len(pos)):
            cv2.rectangle(frame,(int(pos[j][0]-pick[j][2]/2.0) ,int(pos[j][1]-pick[j][3]/2)),(int(pos[j][0]+pick[j][2]/2),int(pos[j][1]+pick[j][3]/2)),(0,255,0),3)
            
        for j in range(len(particles)):
            for i in particles[j]:
                img2 = cv2.circle(frame,(i[0],i[1]),3,255,1)
                
        cv2.imshow('traking_ face',img2)
        k = cv2.waitKey(25) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2) 
        frameCounter = frameCounter + 1


        # rénisiallisé la detection 
        


video = cv2.VideoCapture('people1.mp4')
particle_filtre_traking(video)  

cap.release()
cv2.destroyAllWindows()
