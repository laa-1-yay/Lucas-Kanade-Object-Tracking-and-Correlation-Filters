# Lucas-Kanade-Object-Tracking-and-Correlation-Filters

This project consists of five sections:  
1) A simple Lucas-Kanade (LK) tracker with one single template   
2) The tracker is generalized to accommodate for large appearance variance.   
3) Motion subtraction method for tracking moving pixels in a scene. 
4) Efficient tracking using inverse composition.
5) Efficient tracking using correlation filters.

**1) A simple Lucas-Kanade (LK) tracker with one single template**   

We have a scenario of two-dimensional tracking with a pure translation warp function   
W(x; p) = x + p 

The problem can be described as follows: starting with a rectangular neighborhood of
pixels on frame It, the Lucas-Kanade tracker aims to move it by an offset
p = [px; py]<sup>T</sup> to obtain another rectangle on frame It+1, so that the pixel squared difference
in the two rectangles is minimized:
p∗ = arg minp = X

Starting with an initial guess of p (for instance, p = [0; 0]<sup>T</sup> ), we can
compute the optimal p∗ iteratively. In each iteration, the objective function is locally
linearized by first-order Taylor expansion,

where ∆p = [∆px; ∆py]<sup>T</sup> , is the template offset. 
Further, x0 = W(x; p) = x + p 
is a vector of the x− and y− image gradients at pixel coordinate x0.  
We implement a function with the following signature
```LucasKanade(It, It1, rect, p0 = np.zeros(2))```   
that computes the optimal local motion from frame It to frame It+1 that minimizes Equation 3. Here It is the image frame It, It1 is the image frame It+1, rect is the 4-by-1 vector that represents a rectangle describing all the pixel coordinates within N within the image frame It, and p0 is the initial parameter guess (δx; δy).   
The four components of the rectangle are [x1; y1; x2; y2]<sup>T</sup> , where [x1; y1]<sup>T</sup> is the top-left corner and [x2; y2]<sup>T</sup> is the bottom-right corner. The rectangle is inclusive, i.e., in includes all the four corners. To
deal with fractional movement of the template we can perform interpolation with RectBivariateSpline from the scipy.interpolate package.   

![1](/results/1.png)

The image content we are tracking in the first frame differs from the one in the last frame. This is understandable since we are updating the template after processing each frame and the error can be accumulating. This problem is known as template drifting. There are several ways to mitigate this problem. Iain Matthews et al. (2003, https://www.ri.cmu.edu/publication_view.html?pub_id=4433) suggested one possible approach

New result:

![1](/results/2.png)

**2) Generalized tracker to accommodate for large appearance variance**    

The tracker we have implemented in the first secion, with or without template drifting correction, may suffice if the object being tracked is not subject to drastic appearance variance. However, in real life, this can hardly be the case.

One way to address this issue is to use eigen-space approach (aka, principal component analysis, or PCA). The idea is to analyze the historic data we have collected on the object, and produce a few bases, whose linear combination would most likely to constitute the appearance of the object in the new frame. This is actually similar to the idea of having a lot of templates, but looking for too many templates may be expensive, so we only worry about the principal templates.

Mathematically, suppose we are given a set of k image bases of the same size. We can approximate the appearance variation of the new frame It+1 as a linear combination of the previous frame It and the bases weighted by w = [w1; : : : ; wK]<sup>T</sup> 

![1](/results/3.png)


**3) Motion subtraction method for tracking moving pixels in a scene**    

We implement a tracker for estimating dominant affine motion in a sequence of images and subsequently identify pixels corresponding to moving objects in the scene. 
In the first section, we assumed the the motion is limited to pure translation. In this section we shall implement a tracker for affine motion using a planar affine warp function. To estimate dominant motion, the entire image It will serve as the template to be tracked in image It+1, that is, It+1 is assumed to be approximately an affine warped version of It. This approach is reasonable under the assumption that a majority of the pixels correspond to the stationary objects in the scene whose depth variation is small relative to their distance from the camera.   
Using a planar affine warp function you can recover the vector ∆p = [p1; : : : ; p6]<sup>T</sup>  ,
One can represent this affine warp in homogeneous coordinates as,
x<sub>T</sub>  = M

Note that M will differ between successive image pairs. Starting with an initial guess of p = 0 (i.e. M = I) you will need to solve a sequence of least-squares problem to determine ∆p such that p ! p + ∆p at each iteration. Note that unlike previous examples where the template to be tracked is usually small in comparison with the size of the image.  It will almost always not be contained fully in the warped version It+1. Hence, one must only consider pixels lying in the region common to It and the warped version of It+1 when forming the linear system at each iteration.   
Function with the following signature ```LucasKanadeAffine(It, It1)```  returns the affine transformation matrix M, and It and It1 are It and It+1 respectively.  

Once we compute the transformation matrix M relating an image pair It and It+1, a naive way for determining pixels lying on moving objects is as follows: warp the image It using M so that it is registered to It+1 and subtract it from It+1; the locations where the absolute difference exceeds a threshold can then be declared as corresponding to locations of moving objects. To obtain better results, you can check out the following scipy.morphology functions: binary erosion, and binary dilation.

Function  ```SubtractDominantMotion(image1, image2)```   
where image1 and image2 form the input image pair, returns value mask is a binary image of the same size that dictates which pixels are considered to be corresponding to moving objects. We invoke LucasKanadeAffine in this function to derive the
transformation matrix M, and produce the aforementioned binary mask accordingly.

![4](/results/4.png)

**4) Efficient tracking using inverse composition**    

The inverse composition is more computationally efficient than the classical approach because the
image gradients, jacobian and steepest descent images are computed only once in the beginning for
all the iterations of ∆p. Hence, saving time and calculations.

**5) Efficient tracking using correlation filters**   
![1](/results/5.png)

