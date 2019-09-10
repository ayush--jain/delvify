# delvify

Instructions to run the code:

Run `python run_keras_model.py` with `main_model.h5` in the same directory. From another terminal (which would act as client machine), call the API using curl function. Eg: `curl -X POST -F image=@img.jpg 'http://localhost:5000/predict'` where img.jpg is input. Example files are included in the repo. 

Required packages:
python3+
flask
keras
pillow
tensorflow
numpy
