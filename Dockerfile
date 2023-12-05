FROM eddelbuettel/r2u:22.04

USER root

RUN apt-get update
RUN apt-get -y install pip python3-dev git

RUN pip install --upgrade pip setuptools
RUN pip install --upgrade jupyterlab
RUN pip install expipe tqdm xmltodict
RUN pip install pytest scipy seaborn matplotlib
RUN pip install git+https://github.com/rpy2/rpy2.git
RUN pip install git+https://github.com/CINPLA/spatial-maps.git
RUN R -e "install.packages('IRkernel', options(bspm.sudo = TRUE))"
RUN R -e "IRkernel::installspec(user = FALSE)"

RUN R -e "install.packages('Rtrack', dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('rstatix', dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN pip install openpyxl

RUN groupadd -g 999 jovyan && \
    useradd -r -u 999 -g jovyan jovyan

USER jovyan
WORKDIR /home/jovyan
