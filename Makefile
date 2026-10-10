.PHONY: help train sample chat viz test docs run clean

PYTHON ?= python3

help:
	@$(PYTHON) lab.py

train:
	$(PYTHON) lab.py train --steps 600

sample:
	$(PYTHON) lab.py sample --prompt '问题：'

chat:
	$(PYTHON) lab.py chat

viz:
	$(PYTHON) lab.py viz

test:
	$(PYTHON) lab.py test

docs:
	$(PYTHON) lab.py docs

run:
	$(PYTHON) lab.py run

clean:
	rm -rf __pycache__ docs/.vitepress/dist docs/.vitepress/cache
