FROM public.ecr.aws/lambda/python:3.12

# Install ML dependencies directly into the Lambda task root.
COPY requirements /tmp/requirements
RUN pip install --no-cache-dir -r /tmp/requirements/lambda_ml.txt -t ${LAMBDA_TASK_ROOT}

# Copy shared modules and one lambda service source tree.
COPY shared ${LAMBDA_TASK_ROOT}/shared
ARG SERVICE_NAME
COPY services/${SERVICE_NAME}/src ${LAMBDA_TASK_ROOT}/src

# Lambda image entrypoint handler.
CMD ["src.handler.handler"]
