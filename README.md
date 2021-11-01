# Adaptive_Monitoring_NN

## 1. Install Minikube
 
Follow instruction in https://minikube.sigs.k8s.io/docs/start/
<br />Start minikube with calico plugin

    minikube start --network-plugin=cni --cni=calico
Verify Calico installation

    watch kubectl get pods -l k8s-app=calico-node -A
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

## 6. RabbitMQ 

    Credentials:
    echo "Username      : user"
    echo "Password      : $(kubectl get secret --namespace default rabbitmq -o jsonpath="{.data.rabbitmq-password}" | base64 --decode)"
    echo "ErLang Cookie : $(kubectl get secret --namespace default rabbitmq -o jsonpath="{.data.rabbitmq-erlang-cookie}" | base64 --decode)"
    
To Access the RabbitMQ AMQP port:

    echo "URL : amqp://127.0.0.1:5672/"
    kubectl port-forward --namespace default svc/rabbitmq 5672:5672

To Access the RabbitMQ Management interface:

    echo "URL : http://127.0.0.1:15672/"
    kubectl port-forward --namespace default svc/rabbitmq 15672:15672

To access the RabbitMQ Prometheus metrics, get the RabbitMQ Prometheus URL by running:

    kubectl port-forward --namespace default svc/rabbitmq 9419:9419 &
    echo "Prometheus Metrics URL: http://127.0.0.1:9419/metrics"

## 7. RabbitMQ prometheus metrics
    
    kubectl port-forward --namespace default svc/rabbitmq 9419:9419
    http://127.0.0.1:9419/metrics

## 8. Add RabbitMQ overview dashboard

    https://grafana.com/grafana/dashboards/10991

## 9. Limit minikube node bandwidth

    minikube ssh
    sudo apt-get update
    sudo apt-get install git -y
    sudo ip link add name ifb0 type ifb
    sudo ip link set dev ifb0 up
    git clone https://github.com/magnific0/wondershaper.git
    cd wondershaper/
    sudo ./wondershaper -a eth0 -u 8192 -d 8192

## 10. Run RabbitMQ benchmark

First create a user, e.g. test, in RabbitMQ management (http://127.0.0.1:15672/). 
An example for running the benchmark is the following command:

    docker run -it --net=host --rm pivotalrabbitmq/perf-test:latest -x 1 -y 2 -u "throughput-test-1" -a --id "test 1" -s 400000  --uri amqp://{user}:{pass}@127.0.0.1:5672

For documentation: https://rabbitmq.github.io/rabbitmq-perf-test/stable/htmlsingle/
