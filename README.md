# Deep Learning for Pediatric Bone Age Assessment

Pediatric bone age assessment is a task in radiology that lends itself well to deep learning. The current standard of care is to obtain a radiograph of a patient's left hand and compare it to examplar radiographs from Greulich & Pyle's Radiographic Atlas of Skeletal Development of the Hand and Wrist, which was developed in the 1950s. In essence, this is a pattern matching task, where the radiologist attempts to find the radiograph in the atlas that has the most similar skeletal maturity to that of the patient's radiograph. 

In this repository, you will find code to train a hand detector that crops the region of interest (distal ulna and radius to the most distal phalanx) in order to first standardize the image. Code to train models for bone age assessment are also available. 

Trained models are also provided. There are two sets of trained models: 1) trained on the RSNA Pediatric Bone Age Challenge dataset and 2) trained on a large dataset of pediatric hand radiographs obtained for trauma. 

Inference code to use these models is also provided. 

Further details and instructions for setup and execution to come.