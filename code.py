import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import math


def Task1Task2():
    path = "COMP338_Assignment1_Dataset/Training/"

    ##Task 1
    all_descs = [] #50 imgs, n keypoints, 128 bins

    #Find the paths to all the jpgs and store in list
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if file.endswith(".jpg"):
                files.append(str(os.path.join(r,file)))

    sift = cv2.xfeatures2d.SIFT_create() #Patented version. But free to use in academic setting.

    #Calc keypoints & descriptors
    for x in range(len(files)):
        img_descs = []
        print("Locating Keypoints in: " + files[x])
        img = cv2.imread(files[x], cv2.IMREAD_GRAYSCALE)
        kp, desc = sift.detectAndCompute(img, None)
        #kp, desc = pysift.computeKeypointsAndDescriptors(img)
        for d in desc:
            img_descs.append(d)
        all_descs.append(img_descs)

    ##Task 2:
    cluster_centers = []

    #Cluster into 500 words
    num_words = 500

    #Give cluster pointers random values to begin with
    for y in range(num_words):
        random_img = np.random.randint(0,len(all_descs))
        random_desc = np.random.randint(0,len(all_descs[random_img]))
        cluster_pointer = all_descs[random_img][random_desc]
        cluster_centers.append(cluster_pointer)

    #Repeat n number of times till clusters are positioned well.
    for iteration in range(10):
        print("Iteration: " + str(iteration))
        #Now find out which cluster pointer each description it is closest to.
        descs_dictionary = [] #Put all similar descs in same array.
        for i in range(num_words):
            descs_dictionary.append([]) #Create empty arrays for descs.

        for img in all_descs:
            for desc in img:
                #Find which cluster the description is closest to.
                difs = []
                for cluster in cluster_centers: #Find the euclidian dist between each cluster.
                    diff = np.linalg.norm(cluster - desc)
                    difs.append(diff)
                cloest_index = difs.index(min(difs)) #Which cluster is closest
                descs_dictionary[cloest_index].append(desc)

        #Now move the cluster center to average of each array in descs_dictionary
        new_cluster_centers = []
        for desc_cluster in descs_dictionary:
            running_sum = np.zeros(128)
            for desc in desc_cluster:
                running_sum = np.add(desc, running_sum)
            if(len(desc_cluster)) > 0: #If there was nothing in array, assign new point.
                average = running_sum / len(desc_cluster)
            else: #Assign new random cluster center.
                random_img = np.random.randint(0,len(all_descs))
                random_desc = np.random.randint(0,len(all_descs[random_img]))
                average = all_descs[random_img][random_desc]
            new_cluster_centers.append(average)
        #Update cluster_centers to their new positions.
        cluster_centers = new_cluster_centers;
    np.save('cluster_data.npy', cluster_centers)

#loaded_clusters = np.load('cluster_data.npy')

















            #

def Classify():
    test_path = "COMP338_Assignment1_Dataset/Test/"
    train_path = "COMP338_Assignment1_Dataset/Training/"
    train_dict = []
    def LoadImages(path):
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if file.endswith(".jpg"):
                    files.append(str(os.path.join(r,file)))
        return files

    test_files = LoadImages(test_path)
    train_files = LoadImages(train_path)
    def CalcHistograms(files):
        hists = []
        for f in files:
            img = cv2.imread(f,1)
            col_hists = []
            for i in range(3):
                hist = cv2.calcHist([img],[i],None,[256],[0,256])
                hist/=(250*250) #Normalize the histogram. Sum = 1.
                col_hists.append(hist)
            hists.append(col_hists)
        return hists
    test_hists = CalcHistograms(test_files)
    train_hists = CalcHistograms(train_files)

    #Define funtion to find most common item in list
    def most_frequent(List):
        return max(set(List), key = List.count)

    #Apply K nearest neighbour.
    k=1 #K as 3 gives the highest accuracy. But instructions says "Nearest neighbour" so 1?
    correct = 0 #How many are calssified correctly.
    class_names = [
    "airplanes",
    "cars",
    "dog",
    "faces",
    "keyboard"]
    #Keep track of how accurate it is per class.
    class_guesses_correct = [0,0,0,0,0]

    #For each test image histogram
    for j, test in enumerate(test_hists):
        dists = [] #find the distance to every training img.
        for i, train in enumerate(train_hists, 0):
            dist = 0
            for col in range(3): #For each color channel
                for bin_counter in range(256): #For every bin.
                    #Sum up the squared differences betwen test and training img.
                    dist += (abs(test[col][bin_counter] - train[col][bin_counter]))**2
                dist = math.sqrt(dist)
                #Keep track of dists and training img it was compared against
                dists.append((dist, i))

        dists.sort(key=lambda x: x[0]) #Sorts by 1st element of tuple (dist)
        k_closest = dists[:k]
        k_closest_classes = []
        #Put all the closest distances & index of img it came from into an array.
        for c in k_closest:
            k_closest_classes.append(train_files[c[1]])
        #Pick out the path of the most requent in array. And split on '\'
        guess = str(Path(most_frequent(k_closest_classes)).parent).split('\\')
        true = str(Path(test_files[j]).parent).split('\\')
        #take the last element (the class name) & check if we were correct.
        if(guess[-1] == true[-1]):
            correct += 1
            #Add one to the number correct per class array.
            for m,name in enumerate(class_names):
                if(true[-1] == name):
                    class_guesses_correct[m] += 1
    #Print results
    accuracy = (correct / len(test_files))*100
    print("Overall accuracy: " + str(accuracy) + "%")
    for m,name in enumerate(class_names):
        class_guess[m] = (class_guess[m]/10)*100
        print(name + " accuracy: " + str(class_guesses_correct[m])+ "%")

Classify()
