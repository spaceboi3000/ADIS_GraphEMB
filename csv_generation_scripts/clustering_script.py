@dataclass
class AnalysisConfig:
    """Configuration for clustering analysis.
    
    Attributes:
        base_path: Root directory containing embedding files
        datasets: List of dataset names to analyze
        dimensions: List of embedding dimensions (e.g., ['dim64', 'dim128'])
        output_dir: Directory for saving results and plots (relative to script location)
        embedding_filename: Name of embedding files (default: 'Graph2Vec_embeddings')
        random_state: Random seed for reproducibility
        max_k_clusters: Maximum number of clusters to consider
        n_init_kmeans: Number of K-means initializations
    """
    base_path: str = '.'
    datasets: List[str] = field(default_factory=lambda: ['MUTAG', 'ENZYMES', 'IMDB-MULTI'])
    dimensions: List[str] = field(default_factory=lambda: ['dim64', 'dim128', 'dim256'])
    output_dir: str = './results'
    embedding_filename: str = 'Graph2Vec_embeddings'
    random_state: int = 42
    max_k_clusters: int = 10
    n_init_kmeans: int = 10
    
    def __post_init__(self):
        """Resolve paths relative to script location if they're not absolute."""
        # Get the directory where this script is located
        script_dir = Path(__file__).resolve().parent
        
        # Resolve base_path relative to script location if it's not absolute
        base_path_obj = Path(self.base_path)
        if not base_path_obj.is_absolute():
            base_path_obj = script_dir / base_path_obj
        self.base_path = str(base_path_obj.resolve())
        
        # Resolve output_dir relative to script location if it's not absolute
        output_path = Path(self.output_dir)
        if not output_path.is_absolute():
            output_path = script_dir / output_path
        
        # Create output directories
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / 'plots').mkdir(exist_ok=True)
        
        # Update output_dir to the resolved absolute path
        self.output_dir = str(output_path.resolve())


class EmbeddingLoader:
    """Handles loading of embedding files from disk."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        # Convert base_path to Path object once
        self.base_path = Path(config.base_path)
        
    def load_all(self) -> List[EmbeddingData]:
        """Load all embeddings specified in configuration.
        
        Returns:
            List of EmbeddingData objects
        """
        embeddings_list = []
        
        for dataset in self.config.datasets:
            for dim in self.config.dimensions:
                embedding_data = self._load_single(dataset, dim)
                if embedding_data:
                    embeddings_list.append(embedding_data)
                    print(f"✓ Loaded {dataset}/{dim}: {embedding_data.embeddings.shape}, "
                          f"Labels: {embedding_data.has_labels}")
        
        return embeddings_list
    
    def _load_single(self, dataset: str, dimension: str) -> Optional[EmbeddingData]:
        """Load a single embedding file.
        
        Args:
            dataset: Dataset name
            dimension: Embedding dimension
            
        Returns:
            EmbeddingData object or None if file not found
        """
        # Use the resolved base_path
        base_dir = self.base_path / dataset / dimension
        
        # Try .npy file first
        npy_path = base_dir / f'{self.config.embedding_filename}.npy'
        if npy_path.exists():
            embeddings = np.load(npy_path)
            true_labels = LabelLoader.load_labels(dataset, len(embeddings))
            return EmbeddingData(dataset, dimension, embeddings, true_labels, 'npy')
        
        # Try .csv file
        csv_path = base_dir / f'{self.config.embedding_filename}.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            embeddings = self._extract_embeddings_from_csv(df)
            true_labels = LabelLoader.load_labels(dataset, len(embeddings))
            return EmbeddingData(dataset, dimension, embeddings, true_labels, 'csv')
        
        print(f"⚠ Warning: No embedding file found for {dataset}/{dimension}")
        print(f"  Searched in: {base_dir}")
        return None
    
    @staticmethod
    def _extract_embeddings_from_csv(df: pd.DataFrame) -> np.ndarray:
        """Extract embedding values from CSV DataFrame.
        
        Args:
            df: DataFrame loaded from CSV
            
        Returns:
            Numpy array of embeddings
        """
        # Skip ID column if present
        id_columns = ['graph_id', 'GraphId', 'id', 'ID']
        for col in id_columns:
            if col in df.columns:
                return df.drop(columns=[col]).values
        return df.values


def list_embedding_files(base_path: str, dataset: str, dimension: str):
    """List all files in an embedding directory for diagnostics."""
    script_dir = Path(__file__).resolve().parent
    path_obj = Path(base_path)
    if not path_obj.is_absolute():
        path_obj = script_dir / path_obj
    
    path = path_obj / dataset / dimension
    if path.exists():
        files = list(path.glob('*.npy')) + list(path.glob('*.csv'))
        print(f"📁 Files in {dataset}/{dimension}:")
        for f in files:
            print(f"   - {f.name}")
        return files
    else:
        print(f"❌ Directory doesn't exist: {path}")
        return []


def main():
    """Example usage of the ClusteringAnalyzer."""
    results = []
    
    # Configure analysis of graph2vec
    config1 = AnalysisConfig(
        base_path='../embeddings/embeddings_graph2vec/',
        datasets=['MUTAG', 'ENZYMES', 'IMDB-MULTI'],
        dimensions=['dim64', 'dim128', 'dim256'],
        output_dir='../csvs/clustering_results/graph2vec',
        embedding_filename='Graph2Vec_embeddings',
        random_state=42
    )
    
    # Run analysis 1
    print("\n🔹 Starting Graph2Vec Analysis...")
    analyzer = ClusteringAnalyzer(config1)
    results.append(analyzer.run_analysis())

    # Configure analysis of GIN
    # GIN has complex directory structure with hyperparameters
    # Using specific configurations: gin_dim{X}_ep300_lr0.001_drop0.0_pooladd_gfeat0_norm1_attr1_ls0.05_de0.2
    config2 = AnalysisConfig(
        base_path='../embeddings/embeddings_gin/',
        datasets=['MUTAG', 'ENZYMES', 'IMDB-MULTI'],
        dimensions=[
            'gin_dim64_ep300_lr0.001_drop0.0_pooladd_gfeat0_norm1_attr1_ls0.05_de0.2',
            'gin_dim128_ep300_lr0.001_drop0.0_pooladd_gfeat0_norm1_attr1_ls0.05_de0.2',
            'gin_dim256_ep300_lr0.001_drop0.0_pooladd_gfeat0_norm1_attr1_ls0.05_de0.2'
        ],
        output_dir='../csvs/clustering_results/gin',
        embedding_filename='gin_embeddings',  # Check actual filename in those dirs
        random_state=42
    )
    
    # Run analysis 2
    print("\n🔹 Starting GIN Analysis...")
    analyzer = ClusteringAnalyzer(config2)
    results.append(analyzer.run_analysis())

    # Configure analysis NetLSD
    config3 = AnalysisConfig(
        base_path='../embeddings/embeddings_netlsd/',
        datasets=['MUTAG', 'ENZYMES', 'IMDB-MULTI'],
        dimensions=['dim64', 'dim128', 'dim256'],
        output_dir='../csvs/clustering_results/netlsd',
        embedding_filename='NetLSD_embeddings',
        random_state=42
    )
    
    # Run analysis 3
    print("\n🔹 Starting NetLSD Analysis...")
    analyzer = ClusteringAnalyzer(config3)
    results.append(analyzer.run_analysis())
    
    # Print final summary
    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETED!")
    print("=" * 80)
    print(f"Total embedding sets analyzed: {sum(len(r) if isinstance(r, pd.DataFrame) else 0 for r in results)}")
    
    return results


if __name__ == "__main__":
    main()