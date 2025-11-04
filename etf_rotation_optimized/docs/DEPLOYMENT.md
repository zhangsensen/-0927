# ETF轮动系统 - 部署指南

## 目录

- [系统架构](#系统架构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [本地开发](#本地开发)
- [Kubernetes部署](#kubernetes部署)
- [生产环境部署](#生产环境部署)
- [监控和日志](#监控和日志)
- [故障排除](#故障排除)
- [维护操作](#维护操作)

## 系统架构

ETF轮动系统采用微服务架构，主要组件包括：

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ETF轮动应用   │────│   PostgreSQL    │────│     Redis       │
│   (Python)      │    │    数据库       │    │     缓存        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────┐    │    ┌─────────────────┐
         │   Prometheus    │────┼────│    Grafana      │
         │   监控系统      │         │   可视化面板     │
         └─────────────────┘         └─────────────────┘
         │
         └─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
         │  Elasticsearch  │────│     Kibana      │────│   Filebeat      │
         │   日志存储      │    │   日志可视化     │    │   日志收集      │
         └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 技术栈

- **应用层**: Python 3.11, FastAPI/Flask, Numba, Pandas, NumPy
- **数据层**: PostgreSQL 15, Redis 7
- **容器化**: Docker, Docker Compose
- **编排**: Kubernetes, Helm
- **监控**: Prometheus, Grafana, AlertManager
- **日志**: ELK Stack (Elasticsearch, Kibana, Filebeat)
- **CI/CD**: GitHub Actions, GitHub Container Registry
- **安全**: Trivy, Bandit, OPA/Gatekeeper

## 环境要求

### 最低硬件要求

| 组件 | CPU | 内存 | 存储 | 网络 |
|------|-----|------|------|------|
| 应用Pod | 1核 | 2Gi | 10Gi | 100Mbps |
| PostgreSQL | 0.5核 | 1Gi | 20Gi | 100Mbps |
| Redis | 0.25核 | 256Mi | 5Gi | 100Mbps |
| Prometheus | 0.5核 | 1Gi | 20Gi | 100Mbps |
| Grafana | 0.25核 | 256Mi | 5Gi | 100Mbps |

### 软件依赖

- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.24+
- Helm 3.8+
- kubectl 1.24+

### 开发环境

- Python 3.11+
- Poetry 1.8+
- Git 2.30+

## 快速开始

### 1. 克隆代码

```bash
git clone https://github.com/your-org/etf-rotation-optimized.git
cd etf-rotation-optimized
```

### 2. 本地开发环境

```bash
# 创建环境变量文件
cp .env.example .env

# 启动开发环境
docker-compose -f docker-compose.dev.yml up -d

# 等待服务启动
sleep 30

# 验证服务
curl http://localhost:8080/health
```

### 3. 运行测试

```bash
# 进入应用容器
docker-compose -f docker-compose.dev.yml exec etf-rotation-dev bash

# 安装依赖
poetry install --with dev

# 运行测试
poetry run pytest tests/ -v

# 运行性能测试
poetry run pytest tests/performance/ --benchmark-only
```

## 本地开发

### 开发环境配置

```bash
# 启动完整开发环境
docker-compose -f docker-compose.dev.yml up -d

# 查看服务状态
docker-compose -f docker-compose.dev.yml ps

# 查看日志
docker-compose -f docker-compose.dev.yml logs -f etf-rotation-dev
```

### 热重载开发

```bash
# 开启代码热重载
docker-compose -f docker-compose.dev.yml exec etf-rotation-dev \
  python -m watchdog --patterns="*.py" --recursive --command="echo 'Code changed'" .

# 或者使用文件同步工具
docker-sync start
```

### 数据库管理

```bash
# 连接数据库
docker-compose -f docker-compose.dev.yml exec postgres-dev psql -U etf_user -d etf_rotation_dev

# 运行数据库迁移
docker-compose -f docker-compose.dev.yml exec etf-rotation-dev \
  python scripts/migrate_db.py

# 备份数据库
docker-compose -f docker-compose.dev.yml exec postgres-dev \
  pg_dump -U etf_user etf_rotation_dev > backup.sql
```

### 调试配置

```bash
# 启用调试模式
export DEBUG_MODE=1
export LOG_LEVEL=DEBUG

# 使用调试器
docker-compose -f docker-compose.dev.yml exec etf-rotation-dev \
  python -m debugpy --listen 0.0.0.0:5678 --wait-for-client main.py

# 连接调试器 (VSCode)
# 配置 .vscode/launch.json:
```

```json
{
  "name": "Python: Remote Attach",
  "type": "python",
  "request": "attach",
  "connect": {
    "host": "localhost",
    "port": 5678
  },
  "pathMappings": [
    {
      "localRoot": "${workspaceFolder}",
      "remoteRoot": "/app"
    }
  ]
}
```

## Kubernetes部署

### 1. 准备Kubernetes集群

```bash
# 创建命名空间
kubectl apply -f k8s/namespace.yaml

# 添加Helm仓库
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add elastic https://helm.elastic.co
helm repo update
```

### 2. 配置密钥

```bash
# 安装External Secrets Operator
kubectl apply -f https://raw.githubusercontent.com/external-secrets/external-secrets/main/deploy/kubernetes/manifests/namespace.yaml
kubectl apply -f https://raw.githubusercontent.com/external-secrets/external-secrets/main/deploy/kubernetes/manifests/crds.yaml
kubectl apply -f https://raw.githubusercontent.com/external-secrets/external-secrets/main/deploy/kubernetes/manifests/deployment.yaml

# 创建服务账户
kubectl create serviceaccount etf-rotation-external-secrets-sa -n etf-rotation-prod

# 配置IAM角色 (EKS)
aws iam create-role \
  --role-name etf-rotation-secrets-role \
  --assume-role-policy-document file://trust-policy.json

aws iam attach-role-policy \
  --role-name etf-rotation-secrets-role \
  --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite

# 配置服务账户
kubectl annotate serviceaccount etf-rotation-external-secrets-sa \
  -n etf-rotation-prod \
  eks.amazonaws.com/role-arn=arn:aws:iam::ACCOUNT:role/etf-rotation-secrets-role
```

### 3. 部署应用

```bash
# 使用Helm部署
helm upgrade --install etf-rotation ./helm/etf-rotation \
  --namespace etf-rotation-prod \
  --create-namespace \
  --values ./helm/etf-rotation/values-prod.yaml \
  --wait \
  --timeout=10m

# 或使用kubectl直接部署
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

### 4. 验证部署

```bash
# 检查Pod状态
kubectl get pods -n etf-rotation-prod

# 检查服务状态
kubectl get services -n etf-rotation-prod

# 检查应用健康状态
kubectl port-forward -n etf-rotation-prod service/etf-rotation-service 8080:80
curl http://localhost:8080/health
```

## 生产环境部署

### 1. 环境准备

```bash
# 设置生产环境变量
export NAMESPACE=etf-rotation-prod
export CLUSTER_NAME=etf-rotation-prod
export REGION=us-east-1

# 创建生产环境命名空间
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# 设置标签
kubectl label namespace $NAMESPACE environment=production purpose=etf-rotation
```

### 2. SSL证书配置

```bash
# 安装cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# 创建ClusterIssuer
cat << EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@your-domain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### 3. 蓝绿部署

```bash
# 部署绿色环境
helm upgrade --install etf-rotation-green ./helm/etf-rotation \
  --namespace $NAMESPACE \
  --set deployment.color=green \
  --set service.selectorLabels.version=green \
  --values ./helm/etf-rotation/values-prod.yaml

# 验证绿色环境
kubectl wait --for=condition=ready pod -l app=etf-rotation,color=green \
  -n $NAMESPACE --timeout=10m

# 切换流量
kubectl patch service etf-rotation-service -n $NAMESPACE \
  -p '{"spec":{"selector":{"version":"green"}}}'

# 验证切换
sleep 30
curl https://etf-rotation.your-domain.com/health

# 清理蓝色环境
helm uninstall etf-rotation-blue -n $NAMESPACE || true
```

### 4. 自动扩缩容配置

```bash
# 启用Cluster Autoscaler
kubectl apply -f https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml

# 配置HPA
kubectl apply -f - << EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: etf-rotation-hpa
  namespace: $NAMESPACE
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: etf-rotation-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
EOF
```

## 监控和日志

### 1. 监控系统部署

```bash
# 部署Prometheus
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace etf-rotation-monitoring \
  --create-namespace \
  --values monitoring/prometheus/values-prod.yaml

# 导入Grafana仪表板
kubectl create configmap grafana-dashboards \
  --from-file=monitoring/grafana/dashboards/ \
  -n etf-rotation-monitoring \
  --dry-run=client -o yaml | kubectl apply -f -

# 配置告警规则
kubectl apply -f monitoring/prometheus/rules/
```

### 2. 日志系统部署

```bash
# 部署ELK Stack
helm upgrade --install elasticsearch elastic/elasticsearch \
  --namespace etf-rotation-logging \
  --create-namespace \
  --values monitoring/elasticsearch/values-prod.yaml

helm upgrade --install kibana elastic/kibana \
  --namespace etf-rotation-logging \
  --values monitoring/kibana/values-prod.yaml

helm upgrade --install filebeat elastic/filebeat \
  --namespace etf-rotation-logging \
  --values monitoring/filebeat/values-prod.yaml
```

### 3. 访问监控面板

```bash
# 访问Grafana
kubectl port-forward -n etf-rotation-monitoring svc/grafana 3000:3000
# 浏览器访问: http://localhost:3000
# 用户名: admin, 密码: 查看secret

# 访问Prometheus
kubectl port-forward -n etf-rotation-monitoring svc/prometheus-server 9090:9090
# 浏览器访问: http://localhost:9090

# 访问Kibana
kubectl port-forward -n etf-rotation-logging svc/kibana 5601:5601
# 浏览器访问: http://localhost:5601
```

## 故障排除

### 常见问题

#### 1. Pod启动失败

```bash
# 查看Pod状态
kubectl describe pod -n etf-rotation-prod <pod-name>

# 查看Pod日志
kubectl logs -n etf-rotation-prod <pod-name> --previous

# 查看事件
kubectl get events -n etf-rotation-prod --sort-by=.metadata.creationTimestamp
```

#### 2. 数据库连接问题

```bash
# 测试数据库连接
kubectl run db-test --image=postgres:15 --rm -it -n etf-rotation-prod \
  -- psql -h postgres-service -U etf_user -d etf_rotation -c "SELECT 1;"

# 检查数据库配置
kubectl get secret etf-rotation-db-secrets -n etf-rotation-prod -o yaml

# 检查网络策略
kubectl get networkpolicy -n etf-rotation-prod
```

#### 3. 内存不足

```bash
# 查看资源使用情况
kubectl top pods -n etf-rotation-prod
kubectl top nodes

# 调整资源限制
kubectl patch deployment etf-rotation-app -n etf-rotation-prod -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"etf-rotation","resources":{"limits":{"memory":"6Gi"}}}]}}}}'

# 启用HPA自动扩缩容
kubectl autoscale deployment etf-rotation-app -n etf-rotation-prod \
  --cpu-percent=70 --min=3 --max=10
```

#### 4. 外部API访问问题

```bash
# 测试网络连通性
kubectl run network-test --image=nicolaka/netshoot --rm -it -n etf-rotation-prod \
  -- nslookup api.tushare.pro

# 测试HTTP连接
kubectl run http-test --image=curlimages/curl --rm -it -n etf-rotation-prod \
  -- curl -v https://api.tushare.pro

# 检查DNS配置
kubectl get configmap coredns -n kube-system -o yaml
```

### 调试命令

```bash
# 进入容器调试
kubectl exec -it -n etf-rotation-prod <pod-name> -- bash

# 端口转发
kubectl port-forward -n etf-rotation-prod <pod-name> 8080:8080

# 复制文件
kubectl cp -n etf-rotation-prod <pod-name>:/app/logs ./logs

# 临时Pod调试
kubectl run debug --image=busybox --rm -it -n etf-rotation-prod \
  -- wget -qO- http://etf-rotation-service:8080/health
```

## 维护操作

### 1. 备份和恢复

```bash
# 数据库备份
kubectl exec -n etf-rotation-prod deployment/postgres \
  -- pg_dump -U etf_user etf_rotation > backup-$(date +%Y%m%d).sql

# Redis备份
kubectl exec -n etf-rotation-prod deployment/redis \
  -- redis-cli BGSAVE

# 创建快照
kubectl create namespace backup-$(date +%Y%m%d)
kubectl get all -n etf-rotation-prod -o yaml > etf-rotation-backup-$(date +%Y%m%d).yaml
```

### 2. 滚动更新

```bash
# 更新镜像版本
kubectl set image deployment/etf-rotation-app \
  etf-rotation=ghcr.io/your-org/etf-rotation-optimized:v2.1.0 \
  -n etf-rotation-prod

# 查看更新状态
kubectl rollout status deployment/etf-rotation-app -n etf-rotation-prod

# 回滚到上一版本
kubectl rollout undo deployment/etf-rotation-app -n etf-rotation-prod

# 查看历史版本
kubectl rollout history deployment/etf-rotation-app -n etf-rotation-prod
```

### 3. 配置更新

```bash
# 更新ConfigMap
kubectl create configmap etf-rotation-config \
  --from-file=configs/default.yaml \
  -n etf-rotation-prod \
  --dry-run=client -o yaml | kubectl apply -f -

# 重启Pod应用新配置
kubectl rollout restart deployment/etf-rotation-app -n etf-rotation-prod
```

### 4. 性能优化

```bash
# 调整JVM参数 (如果使用Java组件)
kubectl patch deployment etf-rotation-app -n etf-rotation-prod -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"etf-rotation","env":[{"name":"JAVA_OPTS","value":"-Xms2g -Xmx4g"}]}]}}}}'

# 启用垂直扩缩容
kubectl apply -f - << EOF
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: etf-rotation-vpa
  namespace: etf-rotation-prod
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: etf-rotation-app
  updatePolicy:
    updateMode: "Auto"
EOF
```

### 5. 安全维护

```bash
# 扫描容器漏洞
trivy image ghcr.io/your-org/etf-rotation-optimized:v2.0.0

# 更新基础镜像
docker pull python:3.11-slim
docker build --no-cache -t etf-rotation:latest .

# 审计RBAC权限
kubectl auth can-i --list --as=system:serviceaccount:etf-rotation-prod:etf-rotation-sa

# 检查安全上下文
kubectl get pods -n etf-rotation-prod -o jsonpath='{.items[*].spec.securityContext}'
```

## 应急响应

### 1. 服务中断

```bash
# 快速扩容
kubectl scale deployment etf-rotation-app --replicas=6 -n etf-rotation-prod

# 切换到备用服务
kubectl patch service etf-rotation-service -n etf-rotation-prod \
  -p '{"spec":{"selector":{"version":"blue"}}}'

# 启用维护模式
kubectl patch deployment etf-rotation-app -n etf-rotation-prod \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"etf-rotation","env":[{"name":"MAINTENANCE_MODE","value":"true"}]}]}}}}'
```

### 2. 数据恢复

```bash
# 从备份恢复数据库
kubectl exec -i -n etf-rotation-prod deployment/postgres \
  -- psql -U etf_user etf_rotation < backup-20240101.sql

# 重建缓存
kubectl exec -n etf-rotation-prod deployment/redis \
  -- redis-cli FLUSHALL
```

### 3. 安全事件

```bash
# 隔离受影响的Pod
kubectl cordon <node-name>
kubectl drain <node-name> --ignore-daemonsets --delete-local-data

# 轮换密钥
kubectl delete secret etf-rotation-secrets -n etf-rotation-prod
# External Secrets会自动重建

# 审计日志
kubectl logs -n etf-rotation-prod --since=1h | grep -i error
```

---

## 支持联系

- **运维团队**: devops@your-org.com
- **开发团队**: dev@your-org.com
- **紧急联系**: oncall@your-org.com
- **Slack频道**: #etf-rotation-alerts

## 更多文档

- [API文档](./API.md)
- [架构文档](./ARCHITECTURE.md)
- [安全文档](./SECURITY.md)
- [性能调优指南](./PERFORMANCE.md)