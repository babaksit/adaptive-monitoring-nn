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
    #For disabling the rabbitmq prometheus plugin use the following command
    helm install rabbitmq bitnami/rabbitmq -f config/rabbitmq/values_disabled_prometheus.yaml
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
    export RABBITMQ_PASS=$(kubectl get secret --namespace default rabbitmq -o jsonpath="{.data.rabbitmq-password}" | base64 --decode)
    export RABBITMQ_EC=$(kubectl get secret --namespace default rabbitmq -o jsonpath="{.data.rabbitmq-erlang-cookie}" | base64 --decode)

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
    #set desired bandwidth limit in Kbps e.g. 8192
    sudo ./wondershaper -a eth0 -u 8192 -d 8192

## 9. Run RabbitMQ benchmark

First create a user, e.g. test, in RabbitMQ management (http://127.0.0.1:15672/). 
Follow the instructions stated here: https://github.com/rabbitmq/rabbitmq-perf-test/tree/main/html

An example for json config file could be as follow:

    [{'name':      'consume',
      'type':      'simple',
      'uri': 'amqp://test:test@127.0.0.1:5672',
      'params':    [{'time-limit':     60,
                     'producer-count': 1,
                     'consumer-count': 2,
                     'producer-rate-limit': 2500,
                     'min-msg-size' : 100
                      }]}]
