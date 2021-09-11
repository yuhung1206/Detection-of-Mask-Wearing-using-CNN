# Detection-of-Mask-Wearing-using-CNN

A Convolution Neural Network (CNN) is applied to detect whether the masks are correctly worn.  
  
## Dataset
Over 3000 mask-wearing faces from 682 images were included in this project.  
The images all comes from Medical Masks Dataset, which were originally collected by Cheng Hsun Teng from Eden Social Welfare Foundation at -> https://public.roboflow.com/object-detection/mask-wearing  

  
Our purpose is to classify 3 types of faces, including **correct mask-wearing, wrong mask-wearing & no mask-wearing**  
  
| Type (notation)                    | Num for Train | Num for Test |
| ---------------------------------- | ------------- |------------- |
| Correct mask-wearingl ("Good")     |         2864  |          283 |
| Wrong mask-wearing ("None")        |           104 |           22 |
| No mask-wearing ("Bad")            |          578  |           89 |
  
  
## Execution & Overall Structure of system  
 1. Image Preprocessing : Resize Images to [64,64]  
    ```
    python3 Image_Preprocess.py
    ```
 2. CNN for Classification : training the model with **Torch** package    
    ```
    python3 CNN.py
    ```

## Image Preprocessing  
  1. Extract **faces** (sub-image) from picture according to the bounding box provided by Medical Masks Dataset with **openCV**
  2. Apply **Cubic Spline Interpolation** to resize sub-images to **64 x 64** pixel. 
  3. Shift the mean of pixels from all sub-images to zero.  
    ![image](https://user-images.githubusercontent.com/78803926/132660769-5d42f189-0f19-435e-a9d3-8df4bdb3d6b4.png)
    
## CNN for Classification  
  - Concept of Convolution Neural Network  
    ![image](https://user-images.githubusercontent.com/78803926/132662703-b544ad04-f26c-40ef-83e2-5115992ce4b1.png)
      
  - Structure used in this project  
    ![image](https://user-images.githubusercontent.com/78803926/132662368-35660dbd-b885-4611-84db-b86b8a8ad8d7.png)  
      
  - Imbalanced Dataset Problem  
    
    From the Table given above, it was found that the data amount of "No mask-wearing" is least, which resulted in the poor sensitivity of Test data because the model would focus on the majority class and ignore the minority class.  
      
    | Type (notation)                    | Train Sensitivity | Test Sensitivity |
    | ---------------------------------- | ----------------- |----------------- |
    | Correct mask-wearingl ("Good")     |             99.2% |            96.5% |
    | Wrong mask-wearing ("None")        |             97.4% |            95.5% |
    | No mask-wearing ("Bad")            |             80.8% |              50% |  
    
    To tackle this problem, the heavier penalty is imposed on the minority class to emphasize the learning of the minority class via re-weight scheme.  
    The implmentation is shown as:  
    
    ![image](https://user-images.githubusercontent.com/78803926/132667884-507c7455-61fd-4f03-aa5b-a331197bc49a.png)  
      
  - Resluts after the introduction of re-weight scheme
      
      
    | Type (notation)                    | Train Sensitivity | Test Sensitivity |
    | ---------------------------------- | ----------------- |----------------- |
    | Correct mask-wearingl ("Good")     |             94.3% |            90.1% |
    | Wrong mask-wearing ("None")        |             98.4% |            98.9% |
    | No mask-wearing ("Bad")            |             96.2% |            68.2% |  
    
      



