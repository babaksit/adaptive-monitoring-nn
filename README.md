# Adaptive_Monitoring_NN

## 1. Install Minikube
 
Follow instruction in https://minikube.sigs.k8s.io/docs/start/

## 2. Install prometheus chart

    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add stable https://charts.helm.sh/stable
    helm repo update
    helm install prometheus prometheus-community/kube-prometheus-stack 


## 3. Install rabbitmq chart
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm install rabbitmq bitnami/rabbitmq -f config/rabbitmq/values.yaml


## 4. Prometheus-UI
	kubectl port-forward service/prometheus-kube-prometheus-prometheus 9090
 

## 5. Grafana

    kubectl port-forward deployment/prometheus-grafana 3000

Grafna default values are as follow:
user: admin
pwd: prom-operator

## 6. RabbitMQ manager

    kubectl port-forward --namespace default svc/rabbitmq 15672:15672

## 7. Add RabbitMQ overview dashboard

https://grafana.com/grafana/dashboards/10991





