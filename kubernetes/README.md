# Adaptive_Monitoring_NN

## 1. Install Minikube
 
Follow instruction in https://minikube.sigs.k8s.io/docs/start/
<br />Start minikube with calico plugin
    
    minikube start --nodes 2 --network-plugin=cni --memory 2048 --cpus 4

## 2. Install prometheus chart

    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add stable https://charts.helm.sh/stable
    helm repo update
    helm install prometheus prometheus-community/kube-prometheus-stack -f helm/prometheus/values.yaml
   

## 3. Install rabbitmq chart
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm install rabbitmq bitnami/rabbitmq -f helm/rabbitmq/values.yaml

## 4. Limit minikube second node bandwidth which contains rabbitmq pod

    minikube ssh -n minikube-m02 
    sudo apt-get update
    sudo apt-get install git -y
    sudo ip link add name ifb0 type ifb
    sudo ip link set dev ifb0 up
    git clone https://github.com/magnific0/wondershaper.git
    cd wondershaper/
    #set desired bandwidth limit in Kbps e.g. 8192
    sudo ./wondershaper -a eth0 -u 8192 -d 8192

## 5. Run Workload Simulator
    
    helm install  workload-sim helm/workload_sim/. --set rabbitmqHost=$(kubectl get service/rabbitmq -o jsonpath='{.spec.clusterIP}')
 