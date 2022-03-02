To run MTIX service:
1. Create virtual environment with Python 3.9
2. Change directory to where project is located (inside "mtix")
3. Install required modules:
    
    pip install -r requirements/test.txt
    
4. Install the application itself:
    
    pip install -e .
    
5. Run Django server on the desirable port (here is 7955):
    manage.py runserver 0.0.0.0:7955
6. Do not use "./bin/manage.py", just plain "manage.py", path should be already set up.
7. Service thread will be started after the first health check request. So run
    
    GET <your_server_here>:7955/health
    
8. Reply should look like "200 OK, thread (re)started"
9. Now service is run and listens Kafka topic that is specified in ".env" file, "KFK_CONSUMER_TOPIC" value.
This topic should exist on Kafka servers. The destination (output) Kafka topic name is in "KFK_PRODUCER_TOPIC"
