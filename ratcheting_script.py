import numpy as np
import pandas as pd
import streamlit as st

st.title('Ratcheting Assessment')
st.write('This application shows the N most significant changes in stress values for a particular node between two separate points in time.')

uploaded_file = st.file_uploader("Choose a file")


N = st.text_input("Enter a positive integer value for N:", value="100")

# Validate the input
try:
    N = int(N)
    if N <= 0:
        raise ValueError("N must be a positive integer.")
except ValueError as e:
    st.error(str(e))
    st.stop()


def load_data(filename):
    data = pd.read_csv(filename, delimiter=',', header= None).values

    num_rows, num_col = data.shape

    mask = np.ones(num_col)
    mask[1] = 0
    node_index = data[:,1].astype(int)
    data = data[:,mask.astype(bool)]

    mask2 = np.ones(num_col-1)
    mask2[0] = 0
    times_index = data[:,0]
    data = data[:,mask2.astype(bool)]

    num_nodes = len(np.unique(node_index))
    num_times = len(np.unique(times_index))

    data = data.reshape(num_times, num_nodes, num_col-2)
    times_index = times_index.reshape(num_times, num_nodes)[:,0]

    ti, tj = np.triu_indices(num_times, 1) 
    result = np.abs(data[ti] - data[tj])

    top_indices = np.argpartition(result, kth=-N, axis=None)[-N:]
    orig_indices2 = np.unravel_index(top_indices, result.shape)

    sorted_indices = top_indices[np.argsort(result[orig_indices2])[::-1]]
    orig_indices = np.unravel_index(sorted_indices, result.shape)

    tension_map = {0: "Sigma 11", 1: "Sigma 22", 2: "Sigma 33", 3: "Sigma 12", 4: "Sigma vm", 5: "Sigma 13", 6: "Sigma 23"}

    results = []
    for i in range(N):
        time_sorted_index = orig_indices[0][i]
        node_sorted_index = orig_indices[1][i]
        tension_index = orig_indices[2][i]

        time_range = "{:>10.2f}-{:.2f}".format(
            times_index[ti[time_sorted_index]],
            times_index[tj[time_sorted_index]]
        )
        node_label = "{:10.0f}".format(node_index[node_sorted_index])
        stress_type = "{:10s}".format(tension_map.get(tension_index, 'Unknown value'))
        stress_value = "{:>10.2f}".format(result[time_sorted_index,node_sorted_index, tension_index])

        results.append([time_range, node_label, stress_type, stress_value])

    df = pd.DataFrame(results, columns=['Delta t', 'Node Label', 'Stress Type', 'Stress Value (MPa)'])
    return df

# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

if st.button('Run Analysis'):
    with st.spinner('Processing...'):
        results_df = load_data(uploaded_file)
    st.success('Analysis completed successfully.')
    st.table(results_df)
