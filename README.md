# Adaptive_Monitoring_NN

## 1. Install Minikube
 
Follow instruction in https://minikube.sigs.k8s.io/docs/start/
<br />Start minikube with cni plugin
    
    minikube start --nodes 2 --network-plugin=cni --cpus {desired_cpu_number} --extra-config=kubelet.sync-frequency=1s

## 2. Install prometheus chart

    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add stable https://charts.helm.sh/stable
    helm repo update
    helm install prometheus prometheus-community/kube-prometheus-stack -f helm/prometheus/values.yaml

## 3. Install rabbitmq chart

    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm install rabbitmq bitnami/rabbitmq -f helm/rabbitmq/values.yaml

## 4. Limit worker node network bandwidth 

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
    
    helm install  workload-sim kubernetes/helm/workload_sim/. --set rabbitmqHost=$(kubectl get service/rabbitmq -o jsonpath='{.spec.clusterIP}')

## 6. Run CSV Exporter

    helm install  csv-exporter kubernetes/helm/csv_exporter/. 

## 7. Run a Scenario

    conda activate th
    export PYTHONPATH=.
    nohup python3 rabbitmq/message_handler/scenarios/pub_sub_scenario.py --scenario-config rabbitmq/message_handler/configs/pub_sub_scenario_cit.json --connection-config rabbitmq/message_handler/configs/connection.json &

## 7. Check network usage

    # run this in a script file in the desired node
    #!/bin/bash
    while true
    do
        echo "$(date '+TIME: %H:%M:%S') $(column -t /proc/net/dev)" >> logfile
        sleep 1
    done
 
## 8. Set cluster role 

     kubectl create clusterrolebinding default-cluster-admin --clusterrole=cluster-admin --serviceaccount=default:default

## 9. Install and Run Pipeline as a pod

    helm install pipeline helm/pipeline/.