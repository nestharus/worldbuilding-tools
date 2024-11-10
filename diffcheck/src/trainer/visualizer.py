import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from ast import literal_eval


class SimilarityAnalyzer:
    def __init__(self, threshold_dir='threshold_data'):
        self.threshold_dir = Path(threshold_dir)

        # Load data
        with open(self.threshold_dir / 'thresholds.json', 'r') as f:
            self.thresholds = json.load(f)
        with open(self.threshold_dir / 'stats.json', 'r') as f:
            self.stats = json.load(f)

        # Set style
        plt.style.use('seaborn')

    def parse_relationship(self, rel_str):
        """Convert relationship string back to tuple form."""
        return literal_eval(rel_str)

    def plot_key_relationships_boxplot(self):
        """Create boxplot comparing distributions for key relationship types."""
        # Define key relationships we're interested in
        key_relationships = [
            rel for rel in self.thresholds.keys()
            if any(tag in rel.lower() for tag in ['part', 'aux', 'compound', 'poss'])
        ]

        data = []
        for rel in key_relationships:
            rel_stats = self.stats[rel]
            for corpus, stats in rel_stats.items():
                if corpus != 'combined':  # Skip combined stats
                    quartiles = stats['quartiles']
                    count = stats['count']
                    # Create multiple entries for boxplot
                    data.extend([{
                        'Relationship': rel,
                        'Corpus': corpus,
                        'Similarity': value
                    } for value in np.linspace(quartiles[0], quartiles[2], count)])

        df = pd.DataFrame(data)

        plt.figure(figsize=(15, 8))
        sns.boxplot(x='Relationship', y='Similarity', hue='Corpus', data=df)
        plt.xticks(rotation=45, ha='right')
        plt.title('Similarity Score Distributions for Key Relationships')
        plt.tight_layout()
        plt.savefig(self.threshold_dir / 'key_relationships_boxplot.png')
        plt.close()

    def plot_contraction_analysis(self):
        """Detailed analysis of contraction-related relationships."""
        # Find contraction-related relationships
        contraction_rels = [
            rel for rel in self.thresholds.keys()
            if 'PART' in rel or 'AUX' in rel
        ]

        fig, axes = plt.subplots(2, 1, figsize=(15, 12))

        # Plot 1: Mean similarities across corpora
        means_data = []
        for rel in contraction_rels:
            rel_stats = self.stats[rel]
            for corpus, stats in rel_stats.items():
                if corpus != 'combined':
                    means_data.append({
                        'Relationship': rel,
                        'Corpus': corpus,
                        'Mean Similarity': stats['mean']
                    })

        df_means = pd.DataFrame(means_data)
        sns.barplot(x='Relationship', y='Mean Similarity', hue='Corpus', data=df_means, ax=axes[0])
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
        axes[0].set_title('Mean Similarity Scores for Contraction-Related Relationships')

        # Plot 2: Threshold comparison
        thresholds_data = []
        for rel in contraction_rels:
            thresholds_data.append({
                'Relationship': rel,
                'Threshold': self.thresholds[rel]
            })

        df_thresholds = pd.DataFrame(thresholds_data)
        sns.barplot(x='Relationship', y='Threshold', data=df_thresholds, ax=axes[1])
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
        axes[1].set_title('Calculated Thresholds for Contraction-Related Relationships')

        plt.tight_layout()
        plt.savefig(self.threshold_dir / 'contraction_analysis.png')
        plt.close()

    def plot_corpus_comparison(self):
        """Compare similarity distributions across corpora."""
        corpus_stats = defaultdict(list)

        for rel, rel_stats in self.stats.items():
            for corpus, stats in rel_stats.items():
                if corpus != 'combined':
                    corpus_stats[corpus].extend(
                        np.linspace(stats['min'], stats['max'], stats['count'])
                    )

        plt.figure(figsize=(12, 6))
        for corpus, values in corpus_stats.items():
            sns.kdeplot(data=values, label=corpus)

        plt.title('Similarity Score Distributions by Corpus')
        plt.xlabel('Similarity Score')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.threshold_dir / 'corpus_comparison.png')
        plt.close()

    def plot_relationship_network(self):
        """Create a network visualization of relationship connections."""
        relationships = [self.parse_relationship(rel) for rel in self.thresholds.keys()]

        # Create graph data
        pos_tags = set()
        edges = []
        edge_weights = []

        for rel_str, threshold in self.thresholds.items():
            rel = self.parse_relationship(rel_str)
            pos1, dep1 = rel[0]
            pos2, dep2 = rel[1]
            pos_tags.add(f"{pos1}\n{dep1}")
            pos_tags.add(f"{pos2}\n{dep2}")
            edges.append((f"{pos1}\n{dep1}", f"{pos2}\n{dep2}"))
            edge_weights.append(self.stats[rel_str]['combined']['mean'])

        # Create network plot
        plt.figure(figsize=(15, 15))
        G = nx.Graph()
        G.add_nodes_from(pos_tags)
        G.add_weighted_edges_from([(e[0], e[1], w) for e, w in zip(edges, edge_weights)])

        pos = nx.spring_layout(G, k=1, iterations=50)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                               node_size=2000, alpha=0.7)
        nx.draw_networkx_labels(G, pos, font_size=8)

        # Draw edges with weights as colors
        edges = nx.draw_networkx_edges(G, pos, edge_color=edge_weights,
                                       edge_cmap=plt.cm.viridis,
                                       width=2, alpha=0.5)

        plt.colorbar(edges, label='Mean Similarity Score')
        plt.title('Relationship Network (Edge Color = Mean Similarity)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.threshold_dir / 'relationship_network.png')
        plt.close()

    def generate_summary_report(self):
        """Generate text summary of key findings."""
        report = ["Similarity Analysis Summary", "=" * 25, ""]

        # Overall statistics
        total_relationships = len(self.thresholds)
        total_samples = sum(
            stats['combined']['count']
            for stats in self.stats.values()
        )

        report.extend([
            f"Total Relationships Analyzed: {total_relationships}",
            f"Total Samples: {total_samples}",
            "",
            "Key Relationships:",
            "-" * 15
        ])

        # Analyze key relationships
        for rel_str in sorted(self.thresholds.keys()):
            if any(tag in rel_str.lower() for tag in ['part', 'aux', 'compound', 'poss']):
                stats = self.stats[rel_str]['combined']
                report.extend([
                    f"\nRelationship: {rel_str}",
                    f"Mean: {stats['mean']:.3f}",
                    f"Threshold: {self.thresholds[rel_str]:.3f}",
                    f"Sample Count: {stats['count']}"
                ])

        # Save report
        with open(self.threshold_dir / 'analysis_summary.txt', 'w') as f:
            f.write('\n'.join(report))

    def run_analysis(self):
        """Run all analyses and generate visualizations."""
        self.plot_key_relationships_boxplot()
        self.plot_contraction_analysis()
        self.plot_corpus_comparison()
        self.plot_relationship_network()
        self.generate_summary_report()


if __name__ == '__main__':
    analyzer = SimilarityAnalyzer()
    analyzer.run_analysis()