import os
import json
validation_block = ""
validation_dir = "./validation_plots"

VALIDATION_REPORT_TEMPLATE = "./validation_template.html"
VALIDATION_PLOT_DIRECTORY = "./validation_plots"
VALIDATION_CONFIG_NAME = "config.json"
VALIDATION_PLOT_TEMPLATE_BLOCK_NAME = "validation_plot_block"
VALIDATION_CONFIG_TEMPLATE_BLOCK_NAME = "config_block"
FINAL_REPORT_FILE_NAME = "report.html"

def prepare_order_for_report(Order):
    pass

def compile_validation_report(validation_dir):
    validation_plot_directory = os.listdir(os.path.join(validation_dir, VALIDATION_PLOT_DIRECTORY)
    for plot in validation_plot_directory:
        pathlist = plot.split(".")[1]
        if pathlist[1] = 'png':
            plot_path = os.path.join(validation_dir, VALIDATION_PLOT_DIRECTORY, plot)
            validation_block += f'<img src={plot_path} alt={pathlist[0]}>\n'

    config_location = os.path.join(validation_dir, VALIDATION_CONFIG_NAME)
    with open(config_location, 'r') as initial_config:
        config = json.loads(initial_config.read())
        config_block = json.dumps(config, indent=4)
    
    validation_template_location = os.path.join(validation_dir, VALIDATION_REPORT_TEMPLATE)
    with open(validation_template_location, 'r') as html_template:
        template = Template(html_template.read())
    
    report = template.render({VALIDATION_PLOT_TEMPLATE_BLOCK_NAME: validation_block,
                              VALIDATION_CONFIG_TEMPLATE_BLOCK_NAME: config_block})

    with open(FINAL_REPORT_FILE_NAME, 'w') as final_report:
        final_report.write(report)
        final_report.close()