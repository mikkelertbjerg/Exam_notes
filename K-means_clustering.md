# K-means clustering

Step 1: K = The number of cluster one wants to identify
Step 2: Select 3 random data points
Step 3: Meassure the distance between the 1st point and the 3 initial clusters
Step 4: Assign the 1st point to the nearest cluster
step 5: Calculate the mean of each cluster

Quality can be assesed by adding up the variation within each cluster

Repeat with k new random points.

Repeats for x amount of times with the best result (i.e the one with the "best variance".)

How does one know what to use for K?
The more K increases, the less variance between each cluster, untill k is equal to the number of data points available, i.e hvor cluster per k.
Instead one can calculate the reduction in variation, an elbow plot can help visualize this.

When clustering 2 samples or 2 axes on a diagram or in a x-dimensional space, the euclidean distance is calculated:
```
sqroot x^2 + y^2_
```
When clustering 3 samples or 3 axes on a diagram or in a x-dimensional space, the euclidean distance is calculated:
```
sqroot x^2 + y^2 + z^2
```



## Resources
<ul>
    <li>https://www.youtube.com/watch?v=4b5d3muPQmA&start=56s</li>
    <li>https://en.wikipedia.org/wiki/Euclidean_distance</li>
</ul>