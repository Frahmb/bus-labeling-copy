# MIDA - Initialization
MIDA Website - Last Updated April 2024 by MedAI


## Initialization

To get the system running on your local system, first check settings.py under src/MIDA/
Ensure you have changed the MEDIA_ROOT and CHECKPOINT_ROOT to your systems path to their respective folders

MEDIA_ROOT should connect to your 'media/datasets/' folder:

# '/mnt/c/ExamplePath/media/datasets/'

CHECKPOINT_ROOT should connect to the checkpoints folder found in 'src/MedAI-MAE/checkpoints/':

# '/mnt/c/BryanTestCase/bus-labeling-copy/src/MedAI-MAE/checkpoints/'



*Make sure to complete this step before moving on to hosting the server.* Several migrations are dependent on these values


## Running on Ubuntu


in Ubuntu, direct yourself to the src folder found in the repository. Once inside, make sure to input the following commands:

python3 manage.py makemigrations

python3 manage.py migrate

python3 manage.py createsuperuser 
...go through the username and password process

python3 manage.py init_groups

python3 manage.py init_margin

python3 manage.py runserver
