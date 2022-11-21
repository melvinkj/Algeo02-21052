# **Algeo02-21052**
**Tugas Besar 2 - IF2123 Aljabar Linier dan Geometri**

## **Table of Contents**
* [Program Description](#program_description)
* [How to Run Program](#how_to_run_program)
* [Team Members](#team_members)
* [Job Description](#job_description)
* [Folders and Files Description](#folders_and_files_description)
* [Screenshots](#screenshots)
* [Extras](#extras)

## **Program Description**
*Face recognition* is a biometric technology that is usually used to identify one's face for several importances, especially security. Face recognition program involves a group of stored face images in the database, and based on those face images, the program will learn various form of faces and try to match those learned face images to an image that is being identified. In this project, the face recognition algorithm is constructed using Eigenface.

## **How to Run Program**
1. Clone this repository <br>
`$ git clone https://github.com/melvinkj/Algeo02-21052.git`
2. Change the directory to the location where the main program is stored <br>
`$ cd Algeo02-21052\src`
3. Run the main program <br>
`$ python main.py`

## **Team Members**
<h3> Kelompok 41 - MKJ </h3>
<h4>
<ol>
<li> Melvin Kent Jonathan - 13521052
<li> Juan Christopher Santoso - 13521116
<li> Kandida Edgina Gunawan - 13521155
</ol>
</h4>

## **Job Description**

| Source Code Assignments | Progress Status |
|------------------------ | ----------------|
| **Data Training**  | |
| Image to matrix extraction and image resizing | DONE|
| Grayscale image conversion | DONE |
| Determine mean value of data set's image matrices | DONE |
| Covariance matrix calculation | DONE|
| Eigen values and eigen vectors calculation | DONE|
| **Face Recognizing**  | |
| Eigenfaces calculation | DONE|
| Euclidean distance and similarity calculation  | DONE|
| **Bonus**  | |
| Face detection using camera | DONE|
| Video regarding algorithm explanation and implementation | DONE|

| Report Assignments | Progress Status |
|------------------------ | ----------------|
| Cover, Table of Contents, Table of Images | |
| Chapter 1: Issues Description | DONE|
| Chapter 2: Theoretical Basis | DONE|
| Chapter 3: Program Implementation | |
| Chapter 4: Experimentation and Analysis | |
| Chapter 5: Conclusions and Recommendations | DONE|
| Bibliography, Appendix | |



## **Folders and Files Description**
#### **Folder *src***
1. Folder *assets* <br>
Contains images used on the GUI of the program.
2. File *main.py* <br>
Contains main algorithm of the program.
3. File *submain.py* <br>
Contains main program supporting functions and procedures. 
4. File *backend_proto.py* <br>
Contains all face recognition algorithms.
5. File *cam.py* <br>
Contains face detection algorithm by camera.
#### **Folder *test***
Can be categorized as two aspects. First, the camInput folder contains the face image detected by camera when the program is activated. On the other hand, "training_set" folders contain data set images.
#### **Folder *doc***
Contains the report documentation of the program.


## **Screenshots**
 
## **Extras**
<img src="./src/assets/meme.jpg" alt="Face Recog Meme">