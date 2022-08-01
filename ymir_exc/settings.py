# default location of env yaml file
# if your app is running inside docker image, env file is always at /in/env.yaml
# you can change the location of this file IF AND ONLY IF you are doing test
import os

DEFAULT_ENV_FILE_PATH = os.getenv("DEFAULT_ENV_FILE_PATH", '/in/env.yaml')
