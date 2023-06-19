import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import requests
import seaborn as sns
import folium

# Function to fetch data from api
def fetch_data_from_api(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Failed to fetch data from API.")
        return None

# Function to display matrix graph
def display_matrix_graph(data):
    plt.imshow(data, cmap='viridis')
    plt.colorbar()
    plt.title('Matrix Graph')
    plt.show()

# Function to display bell curve
def display_bell_curve(data):
    frequencies = data.astype(float)  # Convert data to float

    # Compute mean and standard deviation
    mean = np.mean(frequencies)
    std_dev = np.std(frequencies)

    # Generate data points for the bell curve
    x = np.linspace(mean - 3 * std_dev, mean + 3 * std_dev, 100)
    y = norm.pdf(x, mean, std_dev)

    # Plot the bell curve
    plt.plot(x, y)
    plt.xlabel('Data')
    plt.ylabel('Probability Density')
    plt.title('Bell Curve')
    plt.show()

# Function to display colormesh visualization
def display_colormesh(data):
    # Extract x, y, z data from the input
    x = data[:, 0].astype(float)
    y = data[:, 1].astype(float)
    z = data[:, 2].astype(float)

    # Determine the shape of the colormesh data
    n_rows = len(np.unique(y))
    n_cols = len(np.unique(x))

    # Reshape the x, y, z arrays into colormesh grid shape
    X = x.reshape((n_rows, n_cols))
    Y = y.reshape((n_rows, n_cols))
    Z = z.reshape((n_rows, n_cols))

    # Plot the colormesh
    plt.pcolormesh(X, Y, Z, shading='auto')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Colormesh')
    plt.show()

# Function to display scatter plot
def display_scatter_plot(data):
    x = data[:, 0]  # Assuming first column as x-axis data
    y = data[:, 1]  # Assuming second column as y-axis data

    plt.scatter(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot')
    plt.show()

# Function to display bar chart
def display_bar_chart(data):
    x_labels = data[:, 0]  # Assuming first column as x-axis categories
    y_values = data[:, 1]  # Assuming second column as y-axis values

    plt.bar(x_labels, y_values)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Bar Chart')
    plt.show()

# Function to display line graph
def display_line_graph(data):
    x = data[:, 0]  # Assuming first column as x-axis data
    y = data[:, 1]  # Assuming second column as y-axis data

    plt.plot(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Line Graph')
    plt.show()

# Function to display pie chart
def display_pie_chart(data):
    labels = data[:, 0]  # Assuming first column as pie chart labels
    sizes = data[:, 1]  # Assuming second column as corresponding sizes

    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title('Pie Chart')
    plt.show()

# Function to display stats
@staticmethod
def display_statanalysis(data):
    statistics = data.describe()
     # Visualize statistical insights
    print(statistics)

# Function to display geospatial maps
def display_geomaps(data):
    # Convert the NumPy array to a pandas DataFrame
    df = pd.DataFrame(data, columns=['latitude', 'longitude', 'name'])

    # Generate geospatial map
    map = folium.Map(location=[0, 0], zoom_start=2)

    # Add markers to the map
    for _, row in df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        name = row['name']
        folium.Marker([lat, lon], popup=name).add_to(map)

    # Display the map
    map.save('geospatial_map.html')


# Function to display heatmap
def display_heatmap(data):
    heatmap_values = data.values

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_values, cmap='hot')
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Heatmap')
    plt.show()
   
# Function to fetch data from a file
def fetch_data(file_path):
    if file_path.lower().endswith('.txt'):
        data = np.loadtxt(file_path)
    elif file_path.lower().endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.lower() == 'api':
        api_url = input("Enter API URL: ")
        data = fetch_data_from_api(api_url)
        if data is None:
           exit()
    else:
        print("Invalid file format!")
        return None
    return data

def strip_quotes(file_path):
    if file_path.startswith('"') and file_path.endswith('"'):
        return file_path[1:-1]

# User input: file path or manual input
input_type = input("Enter input type ('file' or 'manual' or 'api'): ")

if input_type.lower() == 'file':
    file_path = input("Enter file path : ")
    file_path = strip_quotes(file_path)
    data = fetch_data(file_path)
elif input_type.lower() == 'manual':
    rows = int(input("Enter the number of rows: "))
    columns = int(input("Enter the number of columns: "))
    data = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            data[i, j] = float(input(f"Enter the value for data[{i}][{j}]: "))
elif input_type.lower() == 'api':
    file_path='api'
    data=fetch_data(file_path)
else:
    print("Invalid input type!")
    exit()

# Display options
print("Select visualization:")
print("1. Matrix Graph")
print("2. Bell Curve")
print("3. Scatter Plot")
print("4. Bar Chart")
print("5. Line Graph")
print("6. Pie Chart")
print("7. Colormesh")
print("8. Geospatial map")
print("9. Heatmap")

choice = int(input("Enter your choice: "))

# Perform the selected visualization
if choice == 1:
    display_matrix_graph(data)
    display_statanalysis(data)
elif choice == 2:
    #flattened_data = data.flatten()
    display_bell_curve(data)
    display_statanalysis(data)
elif choice == 3:
    display_scatter_plot(data)
    display_statanalysis(data)
elif choice == 4:
    display_bar_chart(data)
    display_statanalysis(data)
elif choice == 5:
    display_line_graph(data)
    display_statanalysis(data)
elif choice == 6:
    display_pie_chart(data)
    display_statanalysis(data)
elif choice == 7:
    display_colormesh(data)
    display_statanalysis(data)
elif choice == 8:
    display_geomaps(data)
    display_statanalysis(data)
elif choice == 9:
    display_heatmap(data)
    display_statanalysis(data)
else:
    print("Invalid choice!")
