# Table of Contents <!-- omit in toc -->

- [Launching an Instance for Sagemaker Local Mode](#launching-an-instance-for-sagemaker-local-mode)
  - [Defining the Instance Configuration](#defining-the-instance-configuration)
  - [Creating the Instance](#creating-the-instance)
  - [Retrieving the Instance ID and IP Address](#retrieving-the-instance-id-and-ip-address)
  - [Connecting to the Instance](#connecting-to-the-instance)
- [Pushing a Docker Image to ECR](#pushing-a-docker-image-to-ecr)
  - [Docker Instructions](#docker-instructions)
  - [Build and Push the Docker Image](#build-and-push-the-docker-image)
  - [Run the Script](#run-the-script)


# Launching an Instance for Sagemaker Local Mode

In some cases, you may want to run Sagemaker locally. This is especially useful for debugging and testing. In order to do this, you will need to launch an EC2 instance. This section details how to launch an EC2 instance locally. 

## Defining the Instance Configuration
Copy the following into a file called `instance-configuration.json`. Be sure to replace with your name for the Created Key.

<details>
<summary>Instance Configuration JSON</summary>

```json
{
    "MaxCount": 1,
    "MinCount": 1,
    "ImageId": "ami-014e66baad82985a5",
    "InstanceType": "g4dn.xlarge",
    "KeyName": "developer-key",
    "EbsOptimized": true,
    "BlockDeviceMappings": [
        {
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "Encrypted": false,
                "DeleteOnTermination": true,
                "Iops": 3000,
                "SnapshotId": "snap-0392bdfc0e5831bbe",
                "VolumeSize": 50,
                "VolumeType": "gp3",
                "Throughput": 125
            }
        }
    ],
    "NetworkInterfaces": [
        {
            "SubnetId": "subnet-04cefa7e942ab5f89",
            "AssociatePublicIpAddress": true,
            "DeviceIndex": 0,
            "Groups": [
                "sg-0c567cb5bea595f99"
            ]
        }
    ],
    "TagSpecifications": [
        {
            "ResourceType": "instance",
            "Tags": [
                {
                    "Key": "Name",
                    "Value": "Scratchpad"
                },
                {
                    "Key": "Created By",
                    "Value": "Ajay"
                }
            ]
        }
    ],
    "PrivateDnsNameOptions": {
        "HostnameType": "ip-name",
        "EnableResourceNameDnsARecord": true,
        "EnableResourceNameDnsAAAARecord": false
    }
}
```
</details>

## Creating the Instance

Then run the following command to create the instance:

```bash
aws ec2 run-instances --cli-input-json file://instance-configuration.json
```

## Retrieving the Instance ID and IP Address

Use this to retrieve the instance IP address to connect:

```bash
aws ec2 describe-instances --filters "Name=tag:Name,Values=Scratchpad" --query "Reservations[*].Instances[*].[PublicDnsName]" --output text
```

## Connecting to the Instance

Use this to connect to the instance replacing `${PublicDnsName}` with the value retrieved above:

```bash
ssh -i "~/.ssh/developer-key.pem" ec2-user@${PublicDnsName}
``` 

# Pushing a Docker Image to ECR

In the following example, we will push a Docker image to ECR. This Docker image runs the Sagemaker Training Container at `With-Context`

## Docker Instructions

In the repository main directory lives a `Dockerfile`. It contains a specific stage for `Sagemaker`. 

```dockerfile
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0-gpu-py310 as sagemaker
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir wandb dvc"[s3]" && \
    pip freeze
ARG KEY_ID=""
ARG SECRET_KEY=""
ENV AWS_ACCESS_KEY_ID=${KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${SECRET_KEY}
```

## Build and Push the Docker Image

This shell script allows one to push the Docker Image built to ECR. Save this file as `build-and-push.sh` in the repository main directory. 


<details>
<summary> Push Docker Image to ECR </summary>

```shell
image=$1

dockerfile=${2:-Dockerfile}

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

account=$(aws sts get-caller-identity --query Account --output text)
key_id=$3
secret_key=$4
# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=${region:-us-east-1}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"
echo "ECR image fullname: ${fullname}"
# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

# Get the login command from ECR and execute it directly
export _DOCKER_REPO="$(aws ecr get-authorization-token --output text --region "${region}" --query 'authorizationData[].proxyEndpoint')"
aws ecr get-login-password --region "${region}" | docker login -u AWS --password-stdin $_DOCKER_REPO

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build -f ${dockerfile} -t ${image} --target sagemaker . --build-arg REGION=${region} --build-arg KEY_ID=${key_id} --build-arg SECRET_KEY=${secret_key}
docker tag ${image} ${fullname}
docker push ${fullname}
```

</details>

## Run the Script

Run the script with the following command:

```bash
bash build-and-push.sh sagemaker-training-container ${DOCKERFILE} ${AWS_ACCESS_KEY_ID} ${AWS_SECRET_KEY}
```

The AWS Access Key ID and AWS Secret Key can be found in the AWS Console under IAM for DVC Sagemaker User. 

# Debugging Training Code

Debugging training code must be accomplished on an EC2 instance (defined above). The instance **must** be stood up and torn down. Here are some tips to getting started properly. 

## AWS Setup

Start with the pre-configured EC2 profile (specifially with the AMI ID: `ami-014e66baad82985a5`)


## Install Starship and Pytorch

This set up ensures that all future development happens within the Pytorch Conda Environment. 

<details>
<summary> Starship Installation Instructions</summary>

```bash
sudo su root
```

then:

```bash
curl -sS https://starship.rs/install.sh | sh
```

then : 

```bash
exit
```

then copy the following to `~/.bashrc`: 

```bash
echo `eval "$(starship init bash)"` >> ~/.bashrc
echo `eval conda activate pytorch` >> ~/.bashrc
```
</details>

## Teardown

When done debugging, teardown the instance by logging out and then running:

```bash
InstanceId=$(aws ec2 describe-instances --filters "Name=tag:Name,Values=Scratchpad" --query "Reservations[*].Instances[*].[InstanceId]" --output text)
```

Then after selection, run the following command to destroy the instance:

```bash
aws ec2 terminate-instances --instance-ids ${InstanceId}
```