import cv2
import os
import numpy as np

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

#Task1Task2()
loaded_clusters = np.load('cluster_data.npy')

















            #
