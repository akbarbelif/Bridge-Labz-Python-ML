
declare -a pip_packages=('django' 'numpy' 'nose' 'pip' 'matplotlib' 'pandas' 'nose' 'scipy' 'html5lib' 'statsmodels' 'beautifulsoup' 'virtualenv' 'scipy' 'scikit-learn' 'sympy' 'bokeh' 'plotly' 'Gensim' 'Scrapy' 'Statsmodels' 'testtools' 'sklearn' 'scikit-image' 'statsmodels' 'xlrd' 'geopy' 'shapely' 'pyproj' 'xlsxwriter' 'pyip' 'paramiko' 'flake8' 'mpi4py' 'power' 'pytest' 'html5lib' 'pytc' 'pytz' 'python-dateutil' 'setuptools' 'seaborn' 'scipy' '-U tensorflow' 'keras' 'spark' 'requests[security]' 'cryptography' '--upgrade pip' '--upgrade ndg-httpsclient' 'quandl' 'theano')

declare -a apt_get_packages=('oracle-java8-installer' 'build-essential' 'software-properties-common' 'python-dev' 'scala' 'g++' 'libopenblas-dev' 'git' 'spyder' 'ipython' 'ipython-notebook' 'zlibc' 'libssl-dev libffi-dev' 'zlib1g zlib1g-dev')

function essential() {
	figlet MachineLearning
	echo -e " \e[94m (To install all ML packages and libraries)\e[0m"
	sudo apt-get update > /dev/null
	echo -ne '\n' | sudo apt-add-repository ppa:webupd8team/java > /dev/null
	echo -ne '\n' | sudo add-apt-repository ppa:pythonxy/pythonxy-devel > /dev/null
	sudo apt-get update > /dev/null
	sudo python -m easy_install --upgrade pyOpenSSL
	wget https://repo.continuum.io/archive/Anaconda3-4.1.1-Linux-x86_64.sh
	bash Anaconda3-4.1.1-Linux-x86_64.sh
	source .bashrc


   for i in "${apt_get_packages[@]}"
do	
  sudo apt-get install $i
done

 # Check Essential packages install
 if [ $? = 0 ]; then
   echo "Installation Successfully"
 else
   echo "Problems to install Essential packages check this!"
 fi
}

function pip_install() {
  # Install numpy
  output=$(sudo -H pip install $i)
  # Check package install
  if [ $? = 0 ]; then
	if [[ $output = *"Collecting $i"*  ]]; then	
	  echo -e "\e[32m $i Installation Successfully.\e[0m"
	elif [[ $output = *"Requirement already"*  ]]; 		then
	  echo -e "\e[33m $i latest vesion already installed in your machine.\e[0m"
	else
		echo -e "$i error:"
	fi

  else
    echo -e "\e[5m \e[31m Problems to install $i check this!\e[0m $output"
  fi
}


case $1 in	
  all)
   essential
	echo "Loading"
    for i in "${pip_packages[@]}"
	do	
	   pip_install $i
	done
    ;;

    *)
	sudo apt-get install figlet > /dev/null
	figlet MachineLearning
	echo "(To install all ML packages and libraries)"

    echo -e '''	Usage: sh file_name.sh + Option
        \e[94mInitial_MLsetup.sh all \e[0m -> This will install all packages and libraries in Python for ML.'''

esac

