
# Thermal Recovery of Bitumen Project

## Project Information

The project studies and projects a realistic representation of the fluctuations of pressure and temperature within a defined system if thermal recovery operations of hot steam injection and bitumen extraction are carried out. Through calibration, using numerical methods, and uncertainty analysis, a model has been developed which predicts the potential fluctuations of pressure and temperature under different circumstances.

Files in the repository include:  

* main.py: this file when ran will produce all the plots relevant to the study. 
* project_functions.py: this file contains benchmarking and unit tests which are necessary to validating the accuracy of our numerical integration function (solve_ode).  
* model_calibration.py: this file contains the functions for calibrating the model.  
* model_calibration_initial.py: this file contains the functions used for calibrating the initial model.
* model_prediction.py: this file contains the functions for predicting many future scenarios of the model.  
* model_posterior.py: this file contains the functions for the uncertainty analysis of our model.  
* data: this file contains the pilot study's recorded data.
* figures: this file contains all the figures generated from the study.

## Project Use

This project may be used to make recommendation to the applicant in their resource consent application.

## Installation

If you wish to edit, the repository may be cloned and the main.py file may be run using any editor such as Visual Studio Code. 

## Framework Used

The code has been built using Visual Studio Code, under the Python Programming Language.

## Contact

<sjeo598@aucklanduni.ac.nz>  
<ckah285@aucklanduni.ac.nz>  
<dpla864@aucklanduni.ac.nz>

## Contribute

We welcome all pull requests. If you wish to make major changes, please contact us to discuss what you would like to change.

Please ensure tests are updated appropriately.

## Credits

Project Members: Angela Jeong, Chelsea Kah, Denis Plakic.  
Thank you to Dr. David Dempsey and his team for their contributions.


