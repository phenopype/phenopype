YAML
----

"YAML" is a recursive acronym for "YAML Ain't Markup Language". YAML is a human-readable data-serialization language, commonly used for configuration files and in applications where data is being stored or transmitted. It uses both Python-style indentation to indicate nesting (source: https://yaml.org/spec/1.2/spec.html)

YAML specifications

- https://en.wikipedia.org/wiki/YAML
- https://yaml.readthedocs.io/en/latest/overview.html
- https://www.tutorialspoint.com/yaml/yaml_introduction.htm

.. note::
	**Here are the most important rules for YAML syntax:**

		- **indentation rules:**
				- 0 spaces + hyphen + space for modules
				- 4 spaces + hyphen + space in front of functions
				- 8 spaces in front of arguments
		- **separation rules:**
				- modules and functions with arguments are followed by a colon (`:`) and a new line
				- functions without specified arguments don't need a colon
				- arguments are followed by a colon, a space and then the value
		- modules and functions can be emtpy (see`- draw_masks` above), but function arguments *cannot* be emtpy (e.g. `overwrite:` needs to be `true` or `false`)
		- as per Python syntax, optional function arguments can, but don't have to be specified and the functions will just run on default values
		- functions can be added multiple times, but sometimes their output may be overwtritten (e.g. `- threshold` makes sense only once, but `- blur` may be used in multiple locations)
