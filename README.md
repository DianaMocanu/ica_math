###### **Em Algorithm**

In the first part we use the classic em algorithm for obtaining an more accurate position for the clusters.
For storing the information specific to each cluster i have used a class, that has two main methods one for initializing 
with zero, and one for the initial random positioning;

The algorithms used are the ones presented in the lecture for Julia but transposed in Python. Here one of the difference was
that for the division of dd/centroids I had to use the product with the inverse of the matrix because of some python limitations
with broadcast.

The next phase was the improvement of the algorithm:

After the **em** algorithm run for a number of n iterations, then we will we will find out the more jammed clusters (meaning the clusters 
that are closer together) using the **KL** algorithm. Next step will be to calculate the loosest cluster. Meaning the centroid
that is the most apart from its cluster points. For obtaining this value we calculate the average distance between each 
centroid and their cluster.

After we have this two values then we will move the jammed centroid next to the loosest one, updating also its covariance matrix. 
Next we will run again the em algorithm for the updated centroids, but this time we will not reinitialize the clusters values
(probability, covariance matrix and centroid coordinates)

We will do this until there is no change from one iteration to another in terms of the loosest and jammed centroids. Also 
I have added an additional exit solution in case that a maximum number of runs has been reached. The main role of this maximum is in the condition 
that a loop has been observed. 

![image](https://user-images.githubusercontent.com/48009948/102809912-7a0c9880-43cb-11eb-80cb-54beda44e677.png)
![image](https://user-images.githubusercontent.com/48009948/102809993-9dcfde80-43cb-11eb-8f87-3c86e5f03709.png)
![image](https://user-images.githubusercontent.com/48009948/102810038-ae805480-43cb-11eb-98ab-f28a5068ed8a.png)

I have choose to rewrite all the algorithms in python as i have encountered some problems with Julia notebook on my laptop
