sudo yum install python3-pip
sudo pip3 install docker-compose # with root access
sudo systemctl enable docker.service
sudo systemctl start docker.service
sudo systemctl status docker.service

wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

wget https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)

sudo yum install docker
sudo usermod -a -G docker ec2-user
id ec2-user
newgrp docker

sudo docker-compose up -d

