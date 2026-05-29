"""
Graphlet feature extraction and fixed-bin histogramming (nearest-center)
=======================================================================

This module provides three main components:

- Create_Graphlets
    Builds local structural "graphlets" (1-, 2-, and 3-site) from a
    `pymatgen` Structure and derives per-graphlet feature dictionaries
    using an `atomic_features_dict`.

- Graphlet_Analyzer
    Generates data-driven histograms for graphlet features using
    estimated bin widths (Freedman-Diaconis with Sturges fallback)
    and returns both per-bin features and "magpie-like" mean/std features.

- Graphlet_AnalyzerFixedBins2D
    Aggregates graphlet features into histograms using predefined bin
    centers (shape: n_features x n_bins) with a nearest-center counting
    rule. The public interface mirrors the original analyzer.

Notes
-----
- All function and attribute names are preserved exactly as provided.
- Only docstrings and non-functional comments were edited for clarity.
- No executable logic was changed.

Author
------
Krishnanada Mallaya, Yanjun Liu
"""

import numpy as np
from pymatgen.analysis.local_env import VoronoiNN

from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
import matplotlib.pyplot as plt
import re
from scipy.stats import skew, kurtosis
from collections import defaultdict

# Use SpacegroupAnalyzer to get symmetry information
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, PointGroupAnalyzer
from pymatgen.core.structure import Molecule
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


class Create_Graphlets:
    """
    Create graphlets (local structural features) from a CIF/Structure.

    The graphlets include:
    - 1-site graphlets: unique site compositions and counts.
    - 2-site graphlets: bonded pairs with distances and counts.
    - 3-site graphlets: triplets with distances and angles, plus counts.

    Parameters
    ----------
    cif_structure : pymatgen.core.Structure
        Input structure; will be reduced to the primitive cell.
    atomic_radii : dict or None, optional
        Mapping of element symbol to atomic radius (pm). If None,
        radii are derived from `mendeleev`.

    Attributes
    ----------
    structure : pymatgen.core.Structure
        Primitive reduced structure.
    atomic_radii : dict
        Element -> radius (pm).
    neighb_data : dict
        Per-site neighbor information built by `get_neighbors`.
    one_site_graphlets : list of dict
        Set by `Get_1_site_graphlets`.
    two_site_graphlets : list of dict
        Set by `Get_2_site_graphlets`.
    three_site_graphlets : list of dict
        Set by `Get_3_site_graphlets`.
    one_site_features : dict
        Set by `get_features`.
    two_site_features : dict
        Set by `get_features`.
    three_site_features : dict
        Set by `get_features`.
    """
     
    def __init__(self, cif_structure,atomic_radii=None,min_voronoi_weight=1e-2):
        prim_structure=cif_structure.get_primitive_structure() # so we only worry about the smallest unit cell
        self.structure = prim_structure
        self.min_voronoi_weight = float(min_voronoi_weight)
        if(atomic_radii is None):
            from mendeleev import element
            self.atomic_radii = {el.symbol: el.atomic_radius for el in [element(i) for i in range(1, 119)]}
        else:
            self.atomic_radii=atomic_radii
        self.neighb_data=self.get_neighbors()


    
    def remove_non_alphabets(self,input_string):
        """
        Remove non-alphabetic characters from an element label.

        Parameters
        ----------
        input_string : str
            Raw element label.

        Returns
        -------
        str
            Cleaned label containing only letters.
        """
        return re.sub(r'[^a-zA-Z]', '', input_string)
    
    def get_string(self,composition):
        """
        Build a deterministic species string from a composition dict.

        Parameters
        ----------
        composition : dict
            Element (string) -> fraction (float).

        Returns
        -------
        str
            Concatenation of sorted elements and rounded fractions.

        Examples
        --------
        >>> composition = {'Ag': 0.9333, 'Hg': 0.0667}
        >>> Create_Graphlets.get_string(None, composition)
        'Ag0.9333Hg0.0667'
        """
        sorted_elems=sorted(composition.keys())
        species_string=''
        for elem in sorted_elems:
            string=elem+str(np.round(composition[elem],4))
            species_string += string
        return species_string #example:'Ag0.9333Hg0.0667' 
    
    def Max_bond_length(self,composition_1, composition_2,factor=1.5):
        """
        Compute a maximum bond-length threshold (Å) from radii and a scale.

        Parameters
        ----------
        composition_1 : dict
            Element -> fraction for site 1.
        composition_2 : dict
            Element -> fraction for site 2.
        factor : float, optional
            Multiplicative factor applied to the sum of weighted radii.

        Returns
        -------
        float
            Bond-length threshold in Angstrom.
        """
        rad_1=sum([composition_1[e]*self.atomic_radii[self.remove_non_alphabets(e)] for e in composition_1])

        rad_2=sum([composition_2[e]*self.atomic_radii[self.remove_non_alphabets(e)] for e in composition_2])
            
        return (rad_1 + rad_2)*factor/100  # Convert pm to Angstrom
        
    def get_neighbors(self):
        """
        Find nearest neighbors for each site using Voronoi tessellation.

        Notes
        -----
        Raw Voronoi neighbors are first filtered by a minimum Voronoi face
        weight to reject nearly-vanishing periodic contacts before applying
        the distance-based bond cutoff.

        Returns
        -------
        dict
            Mapping site index -> dict with:
            - 'site_composition' : dict
            - 'site_label' : str
            - 'site_coords' : array-like
            - 'neighb_sites' : list
            - 'neighb_coords' : list
            - 'neighb_vectors' : list
            - 'neighb_dists' : list of float
            - 'neighb_compositions' : list of dict
            - 'neighb_labels' : list of str

        Raises
        ------
        ValueError
            If any neighbor distance is below 1 Å.
        """
        voronoi_nn = VoronoiNN()

        
        
        # Find nearest neighbors and store them
        neighb_data=defaultdict(list)
        for i, site in enumerate(self.structure.sites):
            neighbors = voronoi_nn.get_nn_info(self.structure, i)
            site_composition=site.species.as_dict()
            site_label=self.get_string(site_composition)
            site_coords=site.coords
            neighb_sites = []
            neighb_coords = []
            neighb_vectors = []
            neighb_dists=[]
            neighb_labels=[]
            neighb_compositions=[]

            for neighbor in neighbors:
                weight = float(neighbor.get('weight', 0.0))
                if weight < self.min_voronoi_weight:
                    continue
                vector=neighbor['site'].frac_coords-site.frac_coords
                # Adjust vector for periodic boundary conditions
                n_vect = self.structure.lattice.get_cartesian_coords(vector) 

                # Calculate the distance between the atoms
                n_coord = site.coords + n_vect
                n_dist = np.linalg.norm(n_vect)
                n_composition=neighbor['site'].species.as_dict()
                n_label=self.get_string(n_composition)
                if n_dist < 1.:
                    raise ValueError(f"Distance between sites is too small (<1 Å): {n_dist} Å. Site: {site}, Neighbor: {neighbor['site']}")
                if(n_dist<self.Max_bond_length(site_composition,n_composition)):
                    neighb_sites.append(self.structure.sites[neighbor['site_index']])
                    neighb_coords.append(n_coord)
                    neighb_vectors.append(n_vect)
                    neighb_dists.append(n_dist)
                    neighb_compositions.append(n_composition)
                    neighb_labels.append(n_label)
            

            neighb_data[i]={
                'site_composition':site_composition,
                'site_label':site_label,
                'site_coords':site_coords,
                'neighb_sites': neighb_sites,
                'neighb_coords':neighb_coords,
                'neighb_vectors':neighb_vectors,
                'neighb_dists':neighb_dists,
                'neighb_compositions':neighb_compositions,
                'neighb_labels':neighb_labels
                }
        return neighb_data


    def Get_1_site_graphlets(self):
        """
        Generate 1-site graphlets.

        Notes
        -----
        For each unique site composition, a graphlet is created with a count.

        Sets
        ----
        self.one_site_graphlets : list of dict
            Populated with 1-site graphlet records.
        """
        all_sites = defaultdict(dict)
        seen_sites = set()
    
        # Find nearest neighbors and determine local symmetry of each site (there is a better way than this, using wyckoff position)

        for i, site in enumerate(self.structure.sites):
            site_composition=site.species.as_dict()
            site_label=self.get_string(site_composition)
            
            if site_label not in seen_sites:
                seen_sites.add(site_label)
                site_dict = {
                        'g_order':1,  #order of graphlet
                        'atom': site_composition,
                        'count': 1
                    }
                all_sites[site_label]=site_dict
            else:
                all_sites[site_label]['count']+=1   
    
        self.one_site_graphlets=list(all_sites.values())
        return None

    def Get_2_site_graphlets(self):
        """
        Generate 2-site graphlets (bonded pairs).

        Notes
        -----
        For each central-neighbor pair within the bond cutoff, record the
        compositions, the distance, and the aggregated count keyed by
        (sorted labels, rounded distance).

        Sets
        ----
        self.two_site_graphlets : list of dict
            Populated with 2-site graphlet records.
        """
        all_pairs = defaultdict(dict)
        seen_pairs = set()
    
        
        for i, site in enumerate(self.structure.sites):
            neighb_data=self.neighb_data[i]


            site_label=neighb_data['site_label']
            site_composition=neighb_data['site_composition']

            neighb_labels=neighb_data['neighb_labels']
            neighb_compositions=neighb_data['neighb_compositions']
            neighb_dists=neighb_data['neighb_dists']

            for n_label,n_dist,n_composition in zip(neighb_labels,neighb_dists,neighb_compositions):
                pair_tuple = (tuple(sorted([site_label, n_label])), round(n_dist, 1))
                if pair_tuple not in seen_pairs:
                    seen_pairs.add(pair_tuple)
                    pair = {
                        'g_order':2, # order of graphlet
                        'atom': site_composition,
                        'neighbor': n_composition,
                        'distance': n_dist,
                        'count': 1
                    }
                    all_pairs[pair_tuple]=pair
                else:
                    all_pairs[pair_tuple]['count']+=1

        self.two_site_graphlets=list(all_pairs.values())
        return None

        
    


    def Get_3_site_graphlets(self):
        """
        Generate 3-site graphlets (triplets).

        Notes
        -----
        For each center site, all unordered neighbor pairs form a triplet.
        Distances and angles are computed; three distances and three angles
        are sorted. A coarse angle bucket (round(angle_jk / 10) * 10) is
        used in the triplet key. Counts are aggregated over unique keys.

        Sets
        ----
        self.three_site_graphlets : list of dict
            Populated with 3-site graphlet records.
        """
        def calculate_angle(vector1, vector2):
            """
            Compute the angle (degrees) between two vectors.

            Parameters
            ----------
            vector1 : array-like
                First vector.
            vector2 : array-like
                Second vector.

            Returns
            -------
            float
                Angle between vectors in degrees.
            """
            # Calculate the dot product and the magnitudes of the vectors
            dot_prod = np.dot(vector1, vector2)
            cos_angle = dot_prod / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angle_degrees = np.degrees(angle)
            return angle_degrees
    
        
    
        all_triplets = defaultdict(dict)
        seen_triplets = set()
        graphlets_record=set()
        for i, site in enumerate(self.structure.sites):
            neighb_data=self.neighb_data[i]
            
            atom_i=neighb_data['site_composition']
            label_i=neighb_data['site_label']
            coord_i=neighb_data['site_coords']

            neighb_dists=neighb_data['neighb_dists']
            neighb_compositions=neighb_data['neighb_compositions']
            neighb_labels=neighb_data['neighb_labels']
            neighb_vects=neighb_data['neighb_vectors']
            neighb_coords=neighb_data['neighb_coords']
            N_neighbs=len(neighb_labels)
        
            # Loop through all pairs of neighbors to create triplets
            for j in range(N_neighbs):
                dist_j=neighb_dists[j]
                vect_j=neighb_vects[j]
                atom_j=neighb_compositions[j]
                label_j=neighb_labels[j]
                coord_j=neighb_coords[j]
                for k in range(j + 1, N_neighbs):
                    dist_k=neighb_dists[k]
                    vect_k=neighb_vects[k]
                    atom_k=neighb_compositions[k]
                    label_k=neighb_labels[k]
                    coord_k=neighb_coords[k]
                    vect_jk=neighb_vects[j]-neighb_vects[k]
                    dist_jk=np.linalg.norm(vect_jk)
                    #sort the distances between three sites
                    sorted_distances=sorted([dist_j, dist_k, dist_jk])
                    dist_1, dist_2, dist_3 = sorted_distances
                    # Interior angles of the triangle (i, j, k).
                    angle_jk = calculate_angle(vect_j, vect_k)
                    angle_ij = calculate_angle(-vect_k, vect_jk)
                    angle_ik = calculate_angle(vect_j, vect_jk)
                    sorted_angles=sorted([angle_jk, angle_ij, angle_ik])
                    angle_1, angle_2, angle_3 = sorted_angles
                    
            
                    sorted_pairs = sorted([(label_j, dist_j), (label_k, dist_k)], key=lambda x: x[0])
                    sorted_pair_labels = (sorted_pairs[0][0], sorted_pairs[1][0])
                    sorted_pair_dists = (round(sorted_pairs[0][1],1), round(sorted_pairs[1][1],1))
            
                    triplet_tuple = (label_i, sorted_pair_labels,sorted_pair_dists , round(angle_jk/10)*10)
                    
                    tuple_atom_i = tuple(atom_i.items())  # [('C0+', 1.0)]
                    tuple_atom_j = tuple(atom_j.items())
                    tuple_atom_k = tuple(atom_k.items())
                    atom_set = frozenset([tuple_atom_i, tuple_atom_j, tuple_atom_k])  # element identities (unordered)
                    coord_x_set = frozenset([coord_i[0], coord_j[0], coord_k[0]])
                    coord_y_set = frozenset([coord_i[1], coord_j[1], coord_k[1]])
                    coord_z_set = frozenset([coord_i[2], coord_j[2], coord_k[2]])
                    graphlet_key = (atom_set, coord_x_set, coord_y_set, coord_z_set) 
                    
        
                    if (triplet_tuple not in seen_triplets) and (graphlet_key not in graphlets_record):
                        seen_triplets.add(triplet_tuple)
                        graphlets_record.add(graphlet_key)
                        triplet = {
                            'g_order':3,
                            'atom': atom_i, #composition of central atom
                            'pair_1': atom_j, # composition of first neighbor 
                            'pair_2': atom_k, # composition of second neighbor
                            'distance_1': dist_1,
                            'distance_2': dist_2,
                            'distance_3': dist_3,
                            'angle1': angle_1,
                            'angle2': angle_2,
                            'angle3': angle_3,
                            'count': 1
                        }
            
                        all_triplets[triplet_tuple]=triplet
                    elif graphlet_key not in graphlets_record:
                        graphlets_record.add(graphlet_key)
                        all_triplets[triplet_tuple]['count']+=1
        self.three_site_graphlets=list(all_triplets.values())
        return None           









    
    def get_features(self,atomic_features_dict):
        """
        Build per-graphlet feature dictionaries using atomic properties.

        Parameters
        ----------
        atomic_features_dict : dict
            Mapping element symbol -> dict of scalar properties.
            Example: {'Fe': {'Z': 26, 'radius': 126, ...}, ...}

        Sets
        ----
        self.one_site_features : dict
        self.two_site_features : dict
        self.three_site_features : dict
        """
        feat_names=list(list(atomic_features_dict.values())[0].keys())     

        def get_atomic_features(composition): #find average atomic features of the composition
            """
            Compute weighted-average atomic features for a composition.

            Parameters
            ----------
            composition : dict
                Element -> fraction.

            Returns
            -------
            dict
                Feature name -> weighted value.
            """
            composition_feature={}
            for feat in feat_names:
                weighted_feature=sum([float(atomic_features_dict[self.remove_non_alphabets(elem)][feat])*float(occup) 
                                      for elem,occup in composition.items()])
                composition_feature[feat]=weighted_feature 
            return composition_feature                   

        
        try:
            if(self.one_site_graphlets):
                one_G_feats_dict=defaultdict(list)
                for subG in self.one_site_graphlets:
                    atom=subG['atom']
                    count=subG['count']


                    # atomic features
                    atom_feats=get_atomic_features(atom) 
                    for feat in feat_names:
                        one_G_feats_dict[feat+'_1_ord']+=[atom_feats[feat]]*count

                self.one_site_features=one_G_feats_dict
        except:
            pass
        
        try:
            if(self.two_site_graphlets):
                two_G_feat_dict=defaultdict(list)
                
                for subG in self.two_site_graphlets:
                    atom_1=subG['atom']
                    atom_1_feats=get_atomic_features(atom_1)
                    
                    atom_2=subG['neighbor']
                    atom_2_feats=get_atomic_features(atom_2)
                    
                    bond_len=subG['distance']
                    count=subG['count']

                    # atomic features
                    for feat in feat_names:
                        feat_mean=feat+'_mean_2_ord'
                        two_G_feat_dict[feat_mean]+=[np.mean([atom_1_feats[feat],atom_2_feats[feat]])]*count

                        feat_abs=feat+'_abs_2_ord'
                        #two_G_feat_dict[feat_std]+=[np.std([atom_1_feats[feat],atom_2_feats[feat]])]*count
                        two_G_feat_dict[feat_abs]+=[abs(atom_1_feats[feat]-atom_2_feats[feat])]*count
                    two_G_feat_dict['bond_len_2_ord']+=[bond_len]*count

                self.two_site_features=two_G_feat_dict
        except:
            pass


        try:
            if(self.three_site_graphlets):
                three_G_feat_dict=defaultdict(list)
        
                for subG in self.three_site_graphlets:
                    atom=subG['atom']
                    atom_feats=get_atomic_features(atom)

                    atom_1=subG['pair_1']
                    atom_1_feats=get_atomic_features(atom_1)

                    atom_2=subG['pair_2']
                    atom_2_feats=get_atomic_features(atom_2)

                    bond_len_1=subG['distance_1']
                    bond_len_2=subG['distance_2']
                    bond_len_3=subG['distance_3']

                    angle1=subG['angle1']
                    angle2=subG['angle2']
                    angle3=subG['angle3']

                    count=subG['count']


                    # atomic features
                    for feat in feat_names:

                        e0=atom_feats[feat]
                        e1=atom_1_feats[feat]
                        e2=atom_2_feats[feat]
                        
                        feat_mean=feat+'_mean_3_ord'
                        three_G_feat_dict[feat_mean]+=[np.mean([e0,e1,e2])]*count

                        feat_std=feat+'_std_3_ord'
                        three_G_feat_dict[feat_std]+=[np.std([e0,e1,e2])]*count

                        feat_skew=feat+'_skew_3_ord'
                        feat_kurt=feat+'_kurt_3_ord'
                        if (np.std([e0,e1,e2])>1e-6):
                            three_G_feat_dict[feat_skew]+=[skew([e0,e1,e2])]*count
                            three_G_feat_dict[feat_kurt]+=[kurtosis([e0,e1,e2])]*count
                        else:
                            three_G_feat_dict[feat_skew]+=[0.0]*count
                            three_G_feat_dict[feat_kurt]+=[0.0]*count
                    # two site features
                    three_G_feat_dict['bond_len_mean_3_ord']+=[np.mean([bond_len_1,bond_len_2,bond_len_3])]*count
                    three_G_feat_dict['bond_len_std_3_ord']+=[np.std([bond_len_1,bond_len_2,bond_len_3])]*count
                    if (np.std([bond_len_1,bond_len_2,bond_len_3])>1e-6):
                            three_G_feat_dict['bond_len_skew_3_ord']+=[skew([bond_len_1,bond_len_2,bond_len_3])]*count
                            three_G_feat_dict['bond_len_kurt_3_ord']+=[kurtosis([bond_len_1,bond_len_2,bond_len_3])]*count
                    else:
                        three_G_feat_dict['bond_len_skew_3_ord']+=[0.0]*count
                        three_G_feat_dict['bond_len_kurt_3_ord']+=[0.0]*count
                   
                    # three site features
                    three_G_feat_dict['angle_mean_3_ord']+=[np.mean([angle1,angle2,angle3])]*count
                    three_G_feat_dict['angle_std_3_ord']+=[np.std([angle1,angle2,angle3])]*count
                    if (np.std([angle1,angle2,angle3])>1e-6):
                            three_G_feat_dict['angle_skew_3_ord']+=[skew([angle1,angle2,angle3])]*count
                            three_G_feat_dict['angle_kurt_3_ord']+=[kurtosis([angle1,angle2,angle3])]*count
                    else:
                        three_G_feat_dict['angle_skew_3_ord']+=[0.0]*count
                        three_G_feat_dict['angle_kurt_3_ord']+=[0.0]*count
                
                self.three_site_features   = three_G_feat_dict     
        except:
            pass

    @staticmethod
    def _to_builtin(obj):
        """
        Recursively convert numpy-heavy objects to JSON-safe Python types.
        """
        if isinstance(obj, dict):
            return {k: Create_Graphlets._to_builtin(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [Create_Graphlets._to_builtin(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return [Create_Graphlets._to_builtin(v) for v in obj.tolist()]
        if hasattr(obj, "item") and callable(obj.item):
            return obj.item()
        return obj

    def _build_metadata(self, cif_path=None):
        """
        Build a compact metadata block for structure-level JSON payloads.
        """
        metadata = {
            "reduced_formula": self.structure.composition.reduced_formula,
            "lattice_parameters": [float(x) for x in self.structure.lattice.abc],
            "n_sites": int(len(self.structure.sites)),
        }
        if cif_path is not None:
            metadata["cif_path"] = cif_path
        return metadata

    def get_graphlets_dict(self, max_order=3):
        """
        Return graphlet records grouped by order.

        Parameters
        ----------
        max_order : int, optional
            Highest graphlet order to include (1, 2, or 3). Default is 3.

        Returns
        -------
        dict
            Order-grouped graphlet records in JSON-safe form.
        """
        graphlets = {
            "one_site_graphlets": getattr(self, "one_site_graphlets", []),
        }
        if max_order >= 2:
            graphlets["two_site_graphlets"] = getattr(self, "two_site_graphlets", [])
        if max_order >= 3:
            graphlets["three_site_graphlets"] = getattr(self, "three_site_graphlets", [])
        return self._to_builtin(graphlets)

    def get_features_dict(self, max_order=3):
        """
        Return merged feature dictionaries up to the requested graphlet order.
        """
        features_dict = {**getattr(self, "one_site_features", {})}
        if max_order >= 2:
            features_dict.update(getattr(self, "two_site_features", {}))
        if max_order >= 3:
            features_dict.update(getattr(self, "three_site_features", {}))
        return features_dict

    def get_feature_groups(self, max_order=3):
        """
        Return raw feature dictionaries grouped by graphlet order.
        """
        feature_groups = {
            "one_site_features": getattr(self, "one_site_features", {}),
        }
        if max_order >= 2:
            feature_groups["two_site_features"] = getattr(self, "two_site_features", {})
        if max_order >= 3:
            feature_groups["three_site_features"] = getattr(self, "three_site_features", {})
        return self._to_builtin(feature_groups)

    def get_feature_counts(self, max_order=3):
        """
        Return compact value-frequency feature dictionaries grouped by order.
        """
        def feature_to_counts_float(feat_list):
            counts = Counter(float(self._to_builtin(v)) for v in feat_list)
            return sorted([(val, freq) for val, freq in counts.items()], key=lambda x: x[0])

        grouped_features = self.get_feature_groups(max_order=max_order)
        return {
            order_name: {
                feat_name: feature_to_counts_float(values)
                for feat_name, values in feat_dict.items()
            }
            for order_name, feat_dict in grouped_features.items()
        }

    def get_json_payload(
        self,
        cif_path=None,
        max_order=3,
        feature_mode="counts",
        include_graphlets=False,
    ):
        """
        Build a JSON-ready graphlet payload for a single structure.

        Parameters
        ----------
        cif_path : str or None, optional
            Original CIF path to record in metadata.
        max_order : int, optional
            Highest graphlet order to include. Default is 3.
        feature_mode : {"counts", "raw"}, optional
            Whether to include compact value-frequency pairs or raw feature
            value lists. Default is "counts".
        include_graphlets : bool, optional
            If True, include explicit 1/2/3-site graphlet records.

        Returns
        -------
        dict
            JSON-ready structure payload.
        """
        payload = {"metadata": self._build_metadata(cif_path=cif_path)}

        if feature_mode == "counts":
            payload["raw_features_counts"] = self.get_feature_counts(max_order=max_order)
        elif feature_mode == "raw":
            payload["raw_features"] = self.get_feature_groups(max_order=max_order)
        else:
            raise ValueError("feature_mode must be either 'counts' or 'raw'.")

        if include_graphlets:
            payload["graphlets"] = self.get_graphlets_dict(max_order=max_order)

        return payload

    def get_histogram_payload(
        self,
        mode="dynamic",
        max_order=3,
        hist_density=False,
        bin_width_factor=1.0,
        num_bins=None,
        bin_centers_2d=None,
        feature_names=None,
        strict=False,
        verbose=False,
        cif_path=None,
    ):
        """
        Build a JSON-ready histogram payload for this graphlet object.

        Parameters
        ----------
        mode : {"dynamic", "fixed", "fixed_2d"}, optional
            Histogramming strategy. "dynamic" uses learned bin edges from the
            current data via `Graphlet_Analyzer`; "fixed"/"fixed_2d" uses
            `Graphlet_AnalyzerFixedBins2D`. Default is "dynamic".
        max_order : int, optional
            Highest graphlet order to include. Default is 3.
        hist_density : bool, optional
            If True, normalize histogram counts to densities.
        bin_width_factor : float, optional
            Scale factor for dynamic bin width selection.
        num_bins : int or None, optional
            Override bin count for dynamic mode only.
        bin_centers_2d : array-like or None, optional
            Predefined fixed bin centers for fixed 2D mode.
        feature_names : list[str] or None, optional
            Feature names corresponding to rows of `bin_centers_2d`.
        strict : bool, optional
            Enforce exact feature-name coverage for fixed 2D mode.
        verbose : bool, optional
            Enable verbose diagnostics in fixed 2D mode.
        cif_path : str or None, optional
            Original CIF path to record in metadata.

        Returns
        -------
        dict
            JSON-ready histogram payload for a single structure.
        """
        graphlet_list = [self]
        if mode == "dynamic":
            analyzer = Graphlet_Analyzer(
                graphlet_list,
                max_order=max_order,
                bin_width_factor=bin_width_factor,
                hist_density=hist_density,
            )
            hist_names, hist_array, feat_bin_name_list, feat_bin_value_list, feat_magpie_name_list, feat_magpie_value_list = analyzer.get_histogram_features(num_bins=num_bins)
            mode_name = "dynamic"
        elif mode in ("fixed", "fixed_2d"):
            if bin_centers_2d is None or feature_names is None:
                raise ValueError("bin_centers_2d and feature_names are required for fixed 2D histogram mode.")
            analyzer = Graphlet_AnalyzerFixedBins2D(
                graphlet_list,
                bin_centers_2d=bin_centers_2d,
                feature_names=feature_names,
                max_order=max_order,
                hist_density=hist_density,
                verbose=verbose,
                strict=strict,
            )
            hist_names, hist_array, feat_bin_name_list, feat_bin_value_list, feat_magpie_name_list, feat_magpie_value_list = analyzer.get_histogram_features()
            mode_name = "fixed_2d"
        else:
            raise ValueError("mode must be one of: 'dynamic', 'fixed', 'fixed_2d'.")

        return {
            "metadata": self._build_metadata(cif_path=cif_path),
            "histogram_mode": mode_name,
            "hist_names": self._to_builtin(hist_names),
            "hist_array": self._to_builtin(hist_array[0]),
            "feat_bin_names": self._to_builtin(feat_bin_name_list[0]),
            "feat_bin_values": self._to_builtin(feat_bin_value_list[0]),
            "feat_magpie_names": self._to_builtin(feat_magpie_name_list[0]),
            "feat_magpie_values": self._to_builtin(feat_magpie_value_list[0]),
        }

class Graphlet_Analyzer:
    """
    Analyze multiple materials' graphlets and generate histograms.

    Parameters
    ----------
    graphlet_list : list
        List of `Create_Graphlets` objects, one per material.
    max_order : int, optional
        Maximum graphlet order to include (1, 2, or 3). Default is 3.
    bin_width_factor : float, optional
        Scale factor applied to the estimated bin width. Default is 1.0.
    hist_density : bool, optional
        If True, histograms are normalized to density. Default is False.
    """
    def __init__(self, graphlet_list,max_order=3,bin_width_factor=1.0,hist_density=False):
        self.graphlet_list=graphlet_list
        self.max_order=max_order
        self.bin_width_factor=bin_width_factor
        self.hist_density=hist_density
    
    def get_features_dict(self,graphlet):
        """
        Combine feature dictionaries from a single material's graphlet.

        Parameters
        ----------
        graphlet : Create_Graphlets
            Graphlet instance for one material.

        Returns
        -------
        dict
            Keys are feature names; values are lists of scalar values.
        """

        if(self.max_order==3):
            features_dict={**graphlet.one_site_features,
                           **graphlet.two_site_features,
                           **graphlet.three_site_features
                           }
        elif(self.max_order==2):
            features_dict={**graphlet.one_site_features,
                           **graphlet.two_site_features
                           }
                                                                                 
        else:
            features_dict=graphlet.one_site_features
        return features_dict
            
    
    
    def get_bins(self):
        """
        Compute histogram bin edges per feature using Freedman–Diaconis.

        Returns
        -------
        dict
            Mapping feature_name -> np.ndarray of bin edges.

        Notes
        -----
        Uses Freedman–Diaconis rule when IQR > 0; otherwise falls back to a
        Sturges-based width. A minimum width of 0.1 is enforced.
        """

        def calculate_bin_widths(data):
            """
            Calculate the ideal bin width for a histogram (FD/Sturges).

            Parameters
            ----------
            data : array-like
                Sample values for a single feature.

            Returns
            -------
            float
                Computed bin width (after scaling).
            """
            n = len(data)
            if len(np.unique(data)) == 1:
                return 0.1  # if only one data exist, bin width doesnt matter
    
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            # Freedman-Diaconis Rule
            if iqr>0:
                fd_bin_width = 2 * iqr / np.cbrt(n)
            else:
                # using Sturges' rule as a backup
                sturges_bin_width = (max(data) - min(data)) / (np.log2(n) + 1)
                fd_bin_width = sturges_bin_width if sturges_bin_width > 0 else 0.1  
            if fd_bin_width < 0.01:
                fd_bin_width = 0.1 
            return fd_bin_width*self.bin_width_factor
    
        all_features=defaultdict(list)
        for graphlet in self.graphlet_list:

            features_dict=self.get_features_dict(graphlet)

            for feature,values in features_dict.items():
                all_features[feature]+=values
            
            
        bin_ranges={}
        bins={}
        for feature, values in all_features.items():
            
            min_val=np.min(values)
            max_val=np.max(values)
            print(np.array(values).shape)
            bin_width=calculate_bin_widths(values)
            bin_ranges[feature]=(min_val,max_val,bin_width)
            print(f"Feature: {feature}, Min Value: {min_val}, Max Value: {max_val}, Bin Width: {bin_width}")

            bins[feature]=np.arange(min_val-bin_width/2,max_val+3*bin_width/2,bin_width)

        return bins
    
    def get_histogram_features(self,num_bins:int =None):
        """
        Build histograms and magpie-like summaries for all materials.

        Parameters
        ----------
        num_bins : int or None, optional
            If provided, the number of bins to use uniformly per feature.
            If None, use the edges computed by `get_bins()`.

        Returns
        -------
        hist_names : list of str
            Names of histogrammed features.
        hist_array : np.ndarray
            Array of shape (n_samples, n_hists, max_nbins, 2). The last
            axis stores bin midpoints at [:,:,:,0] and bin heights at
            [:,:,:,1]. Unused slots are padded with -1.
        feat_name_list : list of list of str
            Per-sample list of per-bin feature names.
        feat_value_list : list of list of float
            Per-sample list of per-bin values (heights).
        feat_magpie_name_list : list of list of str
            Per-sample list of "magpie-like" feature names (mean/std).
        feat_magpie_value_list : list of list of float
            Per-sample list of mean/std values per feature.
        """

        bins=self.get_bins()
       
        hist_features_dict_list=[] # dict of features, storing the histogram bins, for all materials list
        magpie_features_dict_list=[] # dict of features, with mean and std of the histogram, for all materials list
        
        nbins=[] # to find maximum length of bins of all histograms of all materials
        for graphlet in self.graphlet_list:
            hist_features={}
            magpie_features={}
            
            features_dict=self.get_features_dict(graphlet)
 
            for feature,values in features_dict.items():
                if(num_bins):
                    range=(bins[feature][0],bins[feature][-1])
                    bin_heights,bin_edges=np.histogram(values,bins=num_bins,range=range,density=self.hist_density)
                else:
                    bin_heights,bin_edges=np.histogram(values,bins=bins[feature],density=self.hist_density)
                        
                bin_mids=(bin_edges[0:-1]+bin_edges[1:])*0.5
                hist_features[feature]=(bin_mids,bin_heights)
                nbins.append(len(bin_mids))
                val_mean,val_std=np.mean(values),np.std(values)
                cumulants=[1,2]
                magpie_features[feature]=(cumulants, [val_mean,val_std])
                
            hist_features_dict_list.append(hist_features)
            magpie_features_dict_list.append(magpie_features)
            
        self.hist_features_dict_list=hist_features_dict_list
        self.magpie_features_dict_list=magpie_features_dict_list
        self.max_nbins=max(nbins) 


        feat_bin_name_list=[] # list of bin feature names, for all materials list
        feat_bin_value_list=[] #list of bin feature values, for all materials list
        

        for hist_dict in hist_features_dict_list:
            feat_bin_name=[]
            feat_bin_value=[]
            for feat, (bin_mid,bin_height) in hist_dict.items():
                feat_bin_name+=[feat+'='+str(i) for i in bin_mid]
                feat_bin_value+=list(bin_height)
            
            feat_bin_name_list.append(feat_bin_name)
            feat_bin_value_list.append(feat_bin_value)
        # ensure all feature_bin_names are stored in same order for all materials
        feat_bin_names=feat_bin_name_list[0]
        sorted_feat_bin_value_list = []  # Store the sorted values
        for names, vals in zip(feat_bin_name_list,feat_bin_value_list):
            if names!=feat_bin_names:
                print('feature arrangement changed!!!')
                indices = [names.index(item) for item in feat_bin_names]
                sorted_vals=[vals[i] for i in indices]
                sorted_feat_bin_value_list.append(sorted_vals)
            else:
                sorted_feat_bin_value_list.append(vals)
        feat_bin_name_list=[feat_bin_names]*len(feat_bin_name_list)
        feat_bin_value_list=sorted_feat_bin_value_list

        # create magpie like features (each histogram simplified to mean and std)
        feat_magpie_name_list=[] #list of mean and std names, for all materials list
        feat_magpie_value_list=[] #list of mean and std values, for all materials list
        for magpie_dict in magpie_features_dict_list:
            feat_name=[]
            feat_value=[]
            for feat, (cumulant,mean_std) in magpie_dict.items():
                feat_name+=[feat+'_cumulant='+str(i) for i in cumulant]
                feat_value+=list(mean_std)
            
            feat_magpie_name_list.append(feat_name)
            feat_magpie_value_list.append(feat_value)
        # ensure all feature_magpie_names are stored in same order for all materials
        feat_magpie_names=feat_magpie_name_list[0]
        sorted_feat_magpie_value_list = []  # Store the sorted values
        for names, vals in zip(feat_magpie_name_list,feat_magpie_value_list):
            if names!=feat_magpie_names:
                print('magpie feature arrangement changed!!!')
                indices = [names.index(item) for item in feat_magpie_names]
                sorted_vals=[vals[i] for i in indices]
                sorted_feat_magpie_value_list.append(sorted_vals)
            else:
                sorted_feat_magpie_value_list.append(vals)
        feat_magpie_name_list=[feat_magpie_names]*len(feat_magpie_name_list)
        feat_magpie_value_list=sorted_feat_magpie_value_list



                
        

        # Array to store histogram data (bin midpoints and heights)
        # Shape of the array: (n_samples, n_hists, max_nbins, 2)
        # - n_samples: Number of samples
        # - n_hists: Number of histogram features
        # - max_nbins: Maximum number of bins in any histogram
        #   The extra bins and values to meet the shape, are padded with -1
        # The last dimension of size 2 will store:
        # - bin midpoints at index 0
        # - bin heights at index 1

        self.n_samples=len(hist_features_dict_list)
        hist_names=list(hist_features_dict_list[0].keys()) # names of of each histogram plots
        self.n_hists=len(hist_names)
        hist_array=np.full((self.n_samples,self.n_hists,self.max_nbins,2),-1.0,dtype=float)

        for ns,hist_dict in enumerate(hist_features_dict_list):
            for h_name, (bin_mid,bin_height) in hist_dict.items():
                nh=hist_names.index(h_name)
                nbins=len(bin_mid)
                
                hist_array[ns,nh,0:nbins,0]=bin_mid
                hist_array[ns,nh,0:nbins,1]=bin_height


        return hist_names,hist_array, feat_bin_name_list,feat_bin_value_list,feat_magpie_name_list,feat_magpie_value_list
    

class Graphlet_AnalyzerFixedBins2D:
    """
    Histogram aggregation using predefined bin centers with nearest-center rule.

    Parameters
    ----------
    graphlet_list : list
        List of `Create_Graphlets` objects, one per material.
    bin_centers_2d : np.ndarray
        2D array of shape (n_features, n_bins), containing bin centers for
        each feature. Order is preserved as provided.
    feature_names : list of str
        Names corresponding to the rows of `bin_centers_2d`.
    max_order : int, optional
        Maximum graphlet order to include (1, 2, or 3). Default is 3.
    hist_density : bool, optional
        If True, histograms are normalized to density. Default is False.
    verbose : bool, optional
        If True, enables verbose diagnostics. Default is False.
    strict : bool, optional
        If True, raises if a feature in a sample lacks predefined bins.
        Default is False.

    Notes
    -----
    Counting uses the nearest bin center in absolute difference. Values
    outside the range naturally contribute to the first/last center.
    """
    def __init__(self, graphlet_list, bin_centers_2d, feature_names,
                 max_order=3, hist_density=False, verbose=False, strict=False):
        self.graphlet_list = graphlet_list
        self.max_order = max_order
        self.hist_density = hist_density
        self.verbose = verbose
        self.strict = strict

        bin_centers_2d = np.asarray(bin_centers_2d, dtype=float)
        if bin_centers_2d.ndim != 2:
            raise ValueError("bin_centers_2d must be a 2D array of shape (n_features, n_bins).")
        n_features, _ = bin_centers_2d.shape
        if not isinstance(feature_names, (list, tuple)) or len(feature_names) != n_features:
            raise ValueError("length of feature_names must equal the number of rows in bin_centers_2d.")

        # Preserve the order of provided centers (no sorting)
        self.hist_names = list(feature_names)
        self._centers = [np.array(bin_centers_2d[i, :], dtype=float) for i in range(n_features)]
        self._name_to_row = {name: idx for idx, name in enumerate(self.hist_names)}

    # ---------- Utilities ----------
    @staticmethod
    def _safe_mean_std(values):
        """
        Compute mean/std, safely ignoring NaNs and empty arrays.

        Parameters
        ----------
        values : array-like
            Input values for a single feature.

        Returns
        -------
        mean : float
            Mean of the finite values, or 0.0 if none.
        std : float
            Standard deviation of the finite values, or 0.0 if none.
        """
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            return 0.0, 0.0
        return float(np.mean(values)), float(np.std(values))

    def _get_features_dict(self, graphlet):
        """
        Merge 1/2/3-order feature dictionaries depending on `max_order`.

        Parameters
        ----------
        graphlet : Create_Graphlets
            Graphlet instance for one material.

        Returns
        -------
        dict
            Combined feature dictionary. Empty if attributes are missing.
        """
        if self.max_order == 3:
            return {**getattr(graphlet, "one_site_features", {}),
                    **getattr(graphlet, "two_site_features", {}),
                    **getattr(graphlet, "three_site_features", {})}
        elif self.max_order == 2:
            return {**getattr(graphlet, "one_site_features", {}),
                    **getattr(graphlet, "two_site_features", {})}
        elif self.max_order == 1:
            return getattr(graphlet, "one_site_features", {})
        else:
            return {}

    def _count_on_row(self, values, row_idx):
        """
        Nearest-center counting for a single feature row.

        Parameters
        ----------
        values : array-like
            Raw scalar values for this feature in the sample.
        row_idx : int
            Row index of the feature in `self._centers`.

        Returns
        -------
        counts : np.ndarray
            Count (or density) of assigned values per bin center.
        centers : np.ndarray
            The predefined bin centers corresponding to this feature.
        """
        centers = self._centers[row_idx]
        vals = np.asarray(values, dtype=float).ravel()
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            counts = np.zeros_like(centers, dtype=float)
        else:
            # Compute |value - center| distances and assign to argmin center
            diffs = np.abs(vals[:, None] - centers[None, :])
            idx = np.argmin(diffs, axis=1)
            counts = np.bincount(idx, minlength=len(centers)).astype(float)

        if self.hist_density:
            s = counts.sum()
            if s > 0:
                counts /= s
        return counts, centers

    # ---------- Main API ----------
    def get_histogram_features(self):
        """
        Compute fixed-bin histograms and magpie-like summaries.

        Returns
        -------
        hist_names : list of str
            Feature names in the same order as `feature_names`.
        hist_array : np.ndarray
            Array of shape (n_samples, n_hists, max_nbins, 2). The last
            axis stores bin centers at [:,:,:,0] and counts/densities at
            [:,:,:,1]. Unused slots are filled with -1.
        feat_bin_name_list : list of list of str
            Per-sample list of per-bin names (e.g., f"{feat}={center}").
        feat_bin_value_list : list of list of float
            Per-sample list of per-bin values (counts or densities).
        feat_magpie_name_list : list of list of str
            Per-sample list of mean/std names for each feature.
        feat_magpie_value_list : list of list of float
            Per-sample list of [mean, std] values for each feature.

        Raises
        ------
        KeyError
            If `strict=True` and a sample contains a feature not present
            in `feature_names`.
        """
        n_samples = len(self.graphlet_list)
        n_hists = len(self.hist_names)
        nb_per_hist = [len(c) for c in self._centers] if n_hists > 0 else [0]
        max_nbins = int(max(nb_per_hist)) if nb_per_hist else 0

        hist_array = np.full((n_samples, n_hists, max_nbins, 2), -1.0, dtype=float)
        feat_bin_name_list, feat_bin_value_list = [], []
        feat_magpie_name_list, feat_magpie_value_list = [], []

        for ns, g in enumerate(self.graphlet_list):
            features_dict = self._get_features_dict(g)

            if self.strict:
                unknown_feats = [k for k in features_dict.keys() if k not in self._name_to_row]
                if unknown_feats:
                    raise KeyError(f"The following features have no predefined bins: {unknown_feats}")

            this_bin_names, this_bin_values = [], []
            this_mag_names, this_mag_values = [], []

            for hi, feat in enumerate(self.hist_names):
                row_idx = self._name_to_row[feat]
                values = features_dict.get(feat, [])  # treat missing as empty
                counts, centers = self._count_on_row(values, row_idx)

                nb = len(centers)
                hist_array[ns, hi, :nb, 0] = centers
                hist_array[ns, hi, :nb, 1] = counts

                this_bin_names.extend([f"{feat}={c}" for c in centers])
                this_bin_values.extend(counts.tolist())

                mean_v, std_v = self._safe_mean_std(values)
                this_mag_names.extend([f"{feat}_cumulant=1", f"{feat}_cumulant=2"])
                this_mag_values.extend([mean_v, std_v])

            feat_bin_name_list.append(this_bin_names)
            feat_bin_value_list.append(this_bin_values)
            feat_magpie_name_list.append(this_mag_names)
            feat_magpie_value_list.append(this_mag_values)

        return (
            self.hist_names,
            hist_array,
            feat_bin_name_list,
            feat_bin_value_list,
            feat_magpie_name_list,
            feat_magpie_value_list,
        )
