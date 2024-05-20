pipeline {
    agent any

    environment {
		DOCKERHUB_CREDENTIALS=credentials('dockerhub')
	}
    stages {
        stage('Docker Login') {
            steps {
              sh 'echo $DOCKERHUB_CREDENTIALS_PSW | docker login -u $DOCKERHUB_CREDENTIALS_USR --password-stdin'
            }
        }
        stage('Build & push Docker image') {
            steps {
              sh " ansible-playbook playbook.yaml "
            }
        }
        stage('scan Docker image') {
            steps {
              sh " trivy image --no-progress --severity CRITICAL,HIGH  sami4rhimi/big-data-image:latest"
            }
        }
    }
}