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
        stage ('Clone') {
            git 'https://github.com/samirhimi/bigdata_project.git'
        }
        stage('Build & push Docker image') {
            steps {
              sh " ansible-playbook playbook.yml "
            }
        }
    }
}