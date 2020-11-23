import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import math
import pysift

#Funtion that generates word clusters & saves them to "cluster_data.npy"
def GenerateClusters():
    path = "COMP338_Assignment1_Dataset/Test/"

    ##Task 1
    all_descs = [] #50 imgs, n keypoints, 128 bins

    #Find the paths to all the jpgs and store in list
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if file.endswith(".jpg"):
                files.append(str(os.path.join(r,file)))

    #Un-comment if you value your time.
    #sift = cv2.xfeatures2d.SIFT_create() #Patented version. But free to use in academic setting.

    #Calc keypoints & descriptors
    for x in range(len(files)):
        img_descs = []
        print("Locating Keypoints in: " + files[x])
        img = cv2.imread(files[x], cv2.IMREAD_GRAYSCALE)
        #OpenCV method is SO MUCH FASTER. so uncomment if you dont want to wait 5 hours.
        #kp, desc = pysift.detectAndCompute(img, None)
        kp, desc = pysift.computeKeypointsAndDescriptors(img)
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
    for iteration in range(5):
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
#Uncomment to load cluster data to save training time.
#loaded_clusters = np.load('cluster_data.npy')

def Classify_BagOfWords_KNN(input_k=1):
    #Define general useful things.
    bag_of_words_histograms = np.load('bags_of_words.npy')
    test_hists = bag_of_words_histograms[:50]
    train_hists = bag_of_words_histograms[50:]
    test_path = "COMP338_Assignment1_Dataset/Test/"
    train_path = "COMP338_Assignment1_Dataset/Training/"
    class_names = [
    "airplanes",
    "cars",
    "dog",
    "faces",
    "keyboard"]
    correct = 0 #How many are calssified correctly.
    class_guesses_correct = [0,0,0,0,0] #Keep track of how accurate it is per class.

    #Create Confusion matrix:
    confusion_matrix = []
    for i in range(5): #Make zeros array for each class
        confusion_matrix.append([])
        confusion_matrix[i] = [0,0,0,0,0]

    #Define function that loads images in path.
    def LoadImages(path):
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if file.endswith(".jpg"):
                    files.append(str(os.path.join(r,file)))
        return files
    test_files = LoadImages(test_path)
    train_files = LoadImages(train_path)

    #Define funtion to find most common item in list
    def most_frequent(List):
        return max(set(List), key = List.count)

    #Apply K nearest neighbour algorithm
    k=input_k
    correct_imgs = [] #Store names of imgs that were correctly classified.
    wrong_imgs = []
    for j, test in enumerate(test_hists):    #For each test image histogram
        dists = [] #find the distance to every training img store them here.
        for i, train in enumerate(train_hists, 0):
            dist = 0
            for bin_counter in range(len(test_hists[0])): #For every bin.
                #Sum up the squared differences betwen test and training img.
                dist += (abs(test[bin_counter] - train[bin_counter]))**2
            dist = math.sqrt(dist)
            #Keep track of dists and training img it was compared against
            dists.append((dist, i))
        dists.sort(key=lambda x: x[0]) #Sorts by 1st element of tuple (dist)
        k_closest = dists[:k] #Take the first k elements
        k_closest_classes = []
        #Put all the closest distances & index of img it came from into an array.
        for c in k_closest:
            #Extract the class name;
            name = str(train_files[c[1]]).split('\\')[0]
            k_closest_classes.append(name)
        #Pick out the path of the most requent in array. And split on '\'
        guess = str(Path(most_frequent(k_closest_classes))).split('\\')[-1]
        true = str(Path(test_files[j])).split('\\')[-2]
        true_imgName = (str(Path(test_files[j])).split('\\')[-2]) + \
        "/"+(str(Path(test_files[j])).split('\\')[-1])
        #take the 2nd to last element (the class name) & check if we were correct.
        if(guess == true):
            correct_imgs.append(true_imgName) #Add name of correctly classified img for reference.
            correct += 1
            #Add one to the number correct per class array.
            for m,name in enumerate(class_names):
                if(true == name):
                    class_guesses_correct[m] += 1
        else:
            wrong_imgs.append(true_imgName)

        #Update confusion_matrix.
        x_index = class_names.index(guess)
        y_index = class_names.index(true)
        confusion_matrix[x_index][y_index] += 1

    #Print results
    accuracy = (correct / len(test_files))*100
    print("Overall accuracy: " + str(accuracy) + "%")
    for m,name in enumerate(class_names):
        class_guesses_correct[m] = (class_guesses_correct[m]/10)*100
        print(name + " accuracy: " + str(class_guesses_correct[m])+ "%")
    #Print confusion_matrix
    for it, vec in enumerate(confusion_matrix):
        print(class_names[it] + " " + str(vec))
    #Show imgs
    print("List of images Correctly classified: ")
    print(correct_imgs)
    print("List of images Incorrectly classified: ")
    print(wrong_imgs)

def Classify_BagOfWords_HistogramIntersection(input_k=1):
    #Define general useful things.
    bag_of_words_histograms = np.load('bags_of_words.npy')
    test_hists = bag_of_words_histograms[:50]
    train_hists = bag_of_words_histograms[50:]
    test_path = "COMP338_Assignment1_Dataset/Test/"
    train_path = "COMP338_Assignment1_Dataset/Training/"
    class_names = [
    "airplanes",
    "cars",
    "dog",
    "faces",
    "keyboard"]
    correct = 0 #How many are calssified correctly.
    class_guesses_correct = [0,0,0,0,0] #Keep track of how accurate it is per class.

    #Create Confusion matrix:
    confusion_matrix = []
    for i in range(5): #Make zeros array for each class
        confusion_matrix.append([])
        confusion_matrix[i] = [0,0,0,0,0]

    #Define function that loads images in path.
    def LoadImages(path):
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if file.endswith(".jpg"):
                    files.append(str(os.path.join(r,file)))
        return files
    test_files = LoadImages(test_path)
    train_files = LoadImages(train_path)

    #Define funtion to find most common item in list
    def most_frequent(List):
        return max(set(List), key = List.count)

    def histogram_intersection(h1, h2,bins):
        sm = 0
        for i in range(bins):
            sm += min(h1[i], h2[i])
        return sm


    #Apply K nearest neighbour algorithm
    k = input_k
    correct_imgs = [] #Store names of imgs that were correctly classified.
    wrong_imgs = []
    for j, test in enumerate(test_hists):    #For each test image histogram
        dists = [] #find the distance to every training img store them here.
        for i, train in enumerate(train_hists, 0):
            dist = histogram_intersection(test,train,len(train))
            #Keep track of dists and training img it was compared against
            dists.append((dist, i))
        dists.sort(key=lambda x: x[0]) #Sorts by 1st element of tuple (dist)
        dists.reverse()
        k_closest = dists[:k] #Take the first k elements
        k_closest_classes = []
        #Put all the closest distances & index of img it came from into an array.
        for c in k_closest:
            #Extract the class name;
            name = str(train_files[c[1]]).split('\\')[0]
            k_closest_classes.append(name)
        #Pick out the path of the most requent in array. And split on '\'
        guess = str(Path(most_frequent(k_closest_classes))).split('\\')[-1]
        true = str(Path(test_files[j])).split('\\')[-2]
        true_imgName = (str(Path(test_files[j])).split('\\')[-2]) + \
        "/"+(str(Path(test_files[j])).split('\\')[-1])
        #take the 2nd to last element (the class name) & check if we were correct.
        if(guess == true):
            correct_imgs.append(true_imgName) #Add name of correctly classified img for reference.
            correct += 1
            #Add one to the number correct per class array.
            for m,name in enumerate(class_names):
                if(true == name):
                    class_guesses_correct[m] += 1
        else:
            wrong_imgs.append(true_imgName)

        #Update confusion_matrix.
        x_index = class_names.index(guess)
        y_index = class_names.index(true)
        confusion_matrix[x_index][y_index] += 1

    #Print results
    accuracy = (correct / len(test_files))*100
    print("Overall accuracy: " + str(accuracy) + "%")
    for m,name in enumerate(class_names):
        class_guesses_correct[m] = (class_guesses_correct[m]/10)*100
        print(name + " accuracy: " + str(class_guesses_correct[m])+ "%")
    #Print confusion_matrix
    for it, vec in enumerate(confusion_matrix):
        print(class_names[it] + " " + str(vec))
    #Show imgs
    print("List of images Correctly classified: ")
    print(correct_imgs)
    print("List of images Incorrectly classified: ")
    print(wrong_imgs)

#The followig funtions allow us to compare how accurate it would have been had
#We used the histograms generated by their RGB chanel histograms.
#Function that uses the KNN algorithm to predict the class of test images using colour.
def ClassifyKNN_ColourHistogram():
    #Define general useful things.
    test_path = "COMP338_Assignment1_Dataset/Test/"
    train_path = "COMP338_Assignment1_Dataset/Training/"
    class_names = [
    "airplanes",
    "cars",
    "dog",
    "faces",
    "keyboard"]
    correct = 0 #How many are calssified correctly.
    class_guesses_correct = [0,0,0,0,0] #Keep track of how accurate it is per class.

    #Create Confusion matrix:
    confusion_matrix = []
    for i in range(5): #Make zeros array for each class
        confusion_matrix.append([])
        confusion_matrix[i] = [0,0,0,0,0]

    #Define function that loads images in path.
    def LoadImages(path):
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if file.endswith(".jpg"):
                    files.append(str(os.path.join(r,file)))
        return files

    test_files = LoadImages(test_path)
    train_files = LoadImages(train_path)

    #Define function that calculates the histograms(RGB) of all imgs given
    def CalcHistograms(files):
        hists = []
        for f in files:
            img = cv2.imread(f,1)
            col_hists = []
            for i in range(3): #For the 3 color channels
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

    #Apply K nearest neighbour algorithm
    k=1 #K as 3 gives the highest accuracy. But reports made as k=1
    correct_imgs = []#Store names of imgs that were correctly classified.
    wrong_imgs = []
    for j, test in enumerate(test_hists):    #For each test image histogram
        dists = [] #find the distance to every training img store them here.
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
        guess = str(Path(most_frequent(k_closest_classes))).split('\\')
        true = str(Path(test_files[j])).split('\\')
        #take the 2nd to last element (the class name) & check if we were correct.
        if(guess[-2] == true[-2]):
            correct_imgs.append(true[-1]) #Add name of correctly classified img for reference.
            correct += 1
            #Add one to the number correct per class array.
            for m,name in enumerate(class_names):
                if(true[-2] == name):
                    class_guesses_correct[m] += 1
        else:
            wrong_imgs.append(true[-1])

        #Update confusion_matrix.
        x_index = class_names.index(guess[-2])
        y_index = class_names.index(true[-2])
        confusion_matrix[x_index][y_index] += 1


    #Print results
    accuracy = (correct / len(test_files))*100
    print("Overall accuracy: " + str(accuracy) + "%")
    for m,name in enumerate(class_names):
        class_guesses_correct[m] = (class_guesses_correct[m]/10)*100
        print(name + " accuracy: " + str(class_guesses_correct[m])+ "%")
    #Print confusion_matrix
    print(confusion_matrix)
    #Show imgs
    print(correct_imgs)
    print(wrong_imgs)
#Funtion that uses the Colour histogram intersection of images to predict class of test img.
def ClassifyIntersection_ColourHistogram():
    #Define general useful things.
    test_path = "COMP338_Assignment1_Dataset/Test/"
    train_path = "COMP338_Assignment1_Dataset/Training/"
    class_names = [
    "airplanes",
    "cars",
    "dog",
    "faces",
    "keyboard"]
    correct = 0 #How many are calssified correctly.
    class_guesses_correct = [0,0,0,0,0] #Keep track of how accurate it is per class.

    #Create Confusion matrix:
    confusion_matrix = []
    for i in range(5): #Make zeros array for each class
        confusion_matrix.append([])
        confusion_matrix[i] = [0,0,0,0,0]

    #Define function that loads images in path.
    def LoadImages(path):
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if file.endswith(".jpg"):
                    files.append(str(os.path.join(r,file)))
        return files

    test_files = LoadImages(test_path)
    train_files = LoadImages(train_path)

    #Define function that calculates the histograms(RGB) of all imgs given
    def CalcHistograms(files):
        hists = []
        for f in files:
            img = cv2.imread(f,1)
            col_hists = []
            for i in range(3): #For the 3 color channels
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

    #Apply K nearest neighbour algorithm
    k=1 #K as 3 gives the highest accuracy. But reports made as k=1
    correct_imgs = []#Store names of imgs that were correctly classified.
    wrong_imgs = []
    for j, test in enumerate(test_hists):    #For each test image histogram
        dists = [] #find the distance to every training img store them here.
        for i, train in enumerate(train_hists, 0):
            dist = 0
            for col in range(3): #For each color channel
                dist += cv2.compareHist(test[col], train[col], cv2.HISTCMP_INTERSECT)
                #Keep track of dists and training img it was compared against
            dists.append((dist, i))

        dists.sort(key=lambda x: x[0]) #Sorts by 1st element of tuple (dist)
        dists.reverse()
        k_closest = dists[:k]
        k_closest_classes = []
        #Put all the closest distances & index of img it came from into an array.
        for c in k_closest:
            k_closest_classes.append(train_files[c[1]])
        #Pick out the path of the most requent in array. And split on '\'
        guess = str(Path(most_frequent(k_closest_classes))).split('\\')
        true = str(Path(test_files[j])).split('\\')
        #take the 2nd to last element (the class name) & check if we were correct.
        if(guess[-2] == true[-2]):
            correct_imgs.append(true[-1]) #Add name of correctly classified img for reference.
            correct += 1
            #Add one to the number correct per class array.
            for m,name in enumerate(class_names):
                if(true[-2] == name):
                    class_guesses_correct[m] += 1
        else:
            wrong_imgs.append(true[-1])

        #Update confusion_matrix.
        x_index = class_names.index(guess[-2])
        y_index = class_names.index(true[-2])
        confusion_matrix[x_index][y_index] += 1


    #Print results
    accuracy = (correct / len(test_files))*100
    print("Overall accuracy: " + str(accuracy) + "%")
    for m,name in enumerate(class_names):
        class_guesses_correct[m] = (class_guesses_correct[m]/10)*100
        print(name + " accuracy: " + str(class_guesses_correct[m])+ "%")
    #Print confusion_matrix
    print(confusion_matrix)
    #Show imgs
    print(correct_imgs)
    print(wrong_imgs)

k=8
print("---------oOo---------")
print("k="+str(k))
print("\n")
print("Bag of words KNN classification results: ")
Classify_BagOfWords_KNN(k)
print("\n\n")
print("Bag of words Histogram Intersection results: ")
Classify_BagOfWords_HistogramIntersection(k)






















#
