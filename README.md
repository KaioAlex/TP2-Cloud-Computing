# TP2-Cloud-Computing

## General Goal

In this project, students will design, implement and  deploy a playlist recommendation service built on microservices,  combining a Web front end and a machine learning module.  The service  will be built and tested using continuous integration, and automatically deployed using continuous delivery.  The practice of integrating a  machine learning workflow with DevOps has been referred to as [MLOps](https://neptune.ai/blog/mlops).

You will obtain experience using some of the most popular tools in this context: [Docker](https://www.docker.com/) to containerize application components, [Kubernetes](https://kubernetes.io/) to orchestrate the deployment in a cloud environment, [GitHub](https://github.com) (or another Git provider) as a central code repository, and [ArgoCD](https://argoproj.github.io/cd/) as the continuous delivery framework on top of Kubernetes.

Students will create a recommendation system to recommend playlists to a user  based on a set of songs that the user listened in the past.

Usage example:

- Start the server first:

      $ python3 app.py

- Then, make the requests with the songs:
      
      $ wget --server-response \
         --output-document response.out \
         --header='Content-Type: application/json' \
         --post-data '{"songs": ["Ride Wit Me", "Sweet Emotion"]}' \
         http://localhost:32174/api/recommend

- Check the "response.out" file generated to see 'playlist_ids', 'version' and 'model_datetime':

      $ cat response.out

- To regenerate the model (trained_model.pickle), modify model.py and run the below command to update "trained_model.pickle":

      $ python3 model.py

- Ml Container: Docker pull from DockerHub

      $ docker image pull caioalex/playlists-recommender-ml:0.1

- Frontend Container: Docker pull from DockerHub
  
      $ docker image pull caioalex/playlists-recommender-front:0.1
  
- Run the image:

      $ docker run <ID-caioalex/playlists-recommender-front:0.1>

- Test on browser:

      http://172.17.0.2:32174

- Test via command line (Wget):

      $ wget --server-response \
         --output-document response.out \
         --header='Content-Type: application/json' \
         --post-data '{"songs": ["Ride Wit Me", "Sweet Emotion"]}' \
         http://172.17.0.2:32174/api/recommend
