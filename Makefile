test:
	echo "TODO: Run test suite"

db:
	echo "TODO: start db in background / check if running / etc."
	echo "basically after this runs, we can connect to db"

run:
	echo "TODO: start Expo app"

musicdl:
	source env/bin/activate
	git clone --depth 1 https://github.com/dennisfarmer/musicdl.git
	pip install -e ./musicdl