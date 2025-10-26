"# ADIS_GraphEMB" 

How to download the necessary requirements in a virtual environment 

Do this once: python3 -m venv graphenv 

And then each time: 

1) source graphenv/bin/activate or graphenv\Scripts\activate for Windows 

2) pip install -r requirements.txt 

3) If you have NVIDIA CPU: 

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-2.4.0+cpu.html 

Else if you have NVIDIA GPU: 

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric \ -f https://data.pyg.org/whl/torch-2.4.0+cu121.html 

4) After installing the requirements, verify that everything works correctly: 

```bash python -c "import torch; import torch_geometric; import karateclub; print('Torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| PyG:', torch_geometric.__version__)"