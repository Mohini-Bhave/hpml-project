import matplotlib.pyplot as plt

# Global list to store data points
def plot_model_data(model_name, model_dict,model_data):
    # Append new model data to the global list
    if len(model_data)==0:
        model_data = [(model_dict['speed'], model_dict['accuracy'], model_dict['size'], model_name)]
    else:
        model_data.append((model_dict['speed'], model_dict['accuracy'], model_dict['size'], model_name))
    
    # Clear the current figure to redraw
    #plt.clf()

    # Create the scatter plot
    for speed, accuracy, size, name in model_data:
        plt.scatter(speed, accuracy, s=size, label=name)  # Multiply size by 100 for better visibility

    # Adding labels and title
    plt.xlabel('Speed')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Scatter Plot')
    plt.ylim(0,1)
    plt.legend()

    # Show the plot
    plt.show()
    return model_data

