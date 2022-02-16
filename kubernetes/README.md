# Adaptive_Monitoring_NN

## 1. Install Minikube
 
Follow instruction in https://minikube.sigs.k8s.io/docs/start/
<br />Start minikube with calico plugin
    
    minikube start --nodes 2 --network-plugin=cni --memory 2048 --cpus 2

## 2. Install prometheus chart

    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add stable https://charts.helm.sh/stable
    helm repo update
    helm install prometheus prometheus-community/kube-prometheus-stack -f helm/prometheus/values.yaml

    # For enabling or disabling node exporter or other components of the kube-prometheus-stack you can use the 
    # helm/prometheus/values.yaml and upgrade helm chart with the following command
    
   

## 3. Install rabbitmq chart
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm install rabbitmq bitnami/rabbitmq -f helm/rabbitmq/values.yaml
   
## 4. Export RabbitMQ Parameters

    Credentials:
    echo "Username      : user"
    export RABBITMQ_PASS=$(kubectl get secret --namespace default rabbitmq -o jsonpath="{.data.rabbitmq-password}" | base64 --decode)
    export RABBITMQ_EC=$(kubectl get secret --namespace default rabbitmq -o jsonpath="{.data.rabbitmq-erlang-cookie}" | base64 --decode)

## 5. Install RabbitMQ Exporter

     export RABBITMQ_URL=http://$(kubectl get service/rabbitmq -o jsonpath='{.spec.clusterIP}'):15672
     helm install prometheus-rabbitmq-exporter prometheus-community/prometheus-rabbitmq-exporter -f helm/rabbitmq_exporter/values.yaml --set rabbitmq.url=$RABBITMQ_URL --set rabbitmq.user=user --set rabbitmq.password=$RABBITMQ_PASS 

## 6. Check if RabbitMQ Exporter is being scraped by Prometheus
    
	kubectl port-forward service/prometheus-kube-prometheus-prometheus 9090
Open 127.0.0.1:9090 in browser and check in targets section

## 7. Grafana

    kubectl port-forward deployment/prometheus-grafana 3000

Grafana default values are as follow:
user: admin
pwd: prom-operator


## 8. Limit minikube second node bandwidth which contains rabbitmq pod

    minikube ssh -n minikube-m02 
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

## 10. Run Workload Simulator
    
    helm install  workload-sim helm/workload_sim/. --set rabbitmqHost=$(kubectl get service/rabbitmq -o jsonpath='{.spec.clusterIP}')
