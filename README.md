## ROCKSOLID - MNIST

**Required:**

 - Python 3.6.x
 - Numpy
 - TensorFlow
 - Scikit-learn
 - Matplotlib
 - Seaborn
 - OpenCV

**How to use:**
```mermaid
graph LR
A[RockSolid_MNIST_Model] -- Save --> B((Model.h5))
D[RockSolid_MNIST_LiveScan]
C[TensorBoard or Matplotlib graphs]

B -- Evalutate --> C
C -- Load --> D
C --Reset --> A

![Analyze your model with Matplotlib](https://i.imgur.com/bRtJEVd.jpg)

![View how your model work with webcam](https://i.imgur.com/D4ziqtE.jpg)
