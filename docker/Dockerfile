# Software Copyright Notice
# Copyright 2019 Praelexis (Pty) Ltd.
# All rights are reserved
#
# Copyright exists in this computer program and it is protected by
# copyright law and by international treaties. The unauthorised use,
# reproduction or distribution of this computer program constitute acts
# of copyright infringement and may result in civil and criminal
# penalties. Any infringement will be prosecuted to the maximum extent
# possible.
#
# Praelexis (Pty) Ltd chooses the following address for delivery of all
# legal proceedings and notices:
#    Capital Place F,
#    15-21 Neutron Avenue,
#    Stellenbosch,
#    7600,
#    South Africa.

FROM python:3.6.10

MAINTAINER Daniek Brink  <daniek.brink@gmail.com>

ENV LC_TIME en_ZA.UTF-8

# Create logging folder
RUN mkdir -p /var/log/

# Set default execution directory
WORKDIR /project

# First copy and install the requirements
COPY ./requirements_pip.txt ./requirements_pip.txt
RUN pip install -r ./requirements_pip.txt

# Copy rest of the project and install
COPY . .
RUN pip install .

# Expose the necessary port for the API
EXPOSE 3500

# Default command
CMD ["python", "blueberry_forecasting_api/app.py"]
