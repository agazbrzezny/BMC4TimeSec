For local execution, Docker and Docker Compose are prerequisites.

docker build . -t bmc4timesec  
docker run --rm -p 8080:5000 bmc4timesec
