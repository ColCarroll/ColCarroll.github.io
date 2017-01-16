SSH_USER = lyra
DOCUMENT_ROOT = ~/colinweb
PUBLIC_DIR = public/

all: deploy

server:
	hugo server -w .

deploy: site
	rsync -rav $(PUBLIC_DIR) $(SSH_USER):$(DOCUMENT_ROOT)

site: .FORCE
	hugo
	find public -type d -print0 | xargs -0 chmod 755
	find public -type f -print0 | xargs -0 chmod 644

clean:
	rm -rf public/

.FORCE:
