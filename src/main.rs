use std::{fs::File, io::Write};

use linfa::{Dataset, traits::Fit};
use linfa_trees::{DecisionTree, SplitQuality};
use ndarray::{Array2, Axis, array, s};

fn main() {
    let original_data: Array2<f32> = array!(
        [60.,   1.,     12.,    13.,     10.],
        [30.,   2.,     3.,     13.,     4.],
        [15.,   5.,     1.,     9.,      8.],
        [5.,    1.,     6.,     9.,      4.],
        [22.,   8.,     6.,     10.,     1.],
        [24.,   2.,     6.,     16.,     1.],
        [50.,   4.,     12.,    13.,     10.],
        [40.,   6.,     3.,     11.,     4.],
        [35.,   5.,     1.,     9.,      8.],
        [52.,   7.,     4.,     8.,      9.],
        [21.,   8.,     4.,     10.,     1.],
        [31.,   3.,     3.,     11.,     8.]
    );
    let feature_names = vec!["ConsultLength", "Consultant", "ConsultLocation", "HourOfDay", "ConsultRating"];
    let num_features = original_data.len_of(Axis(1)) - 1;
    let features = original_data.slice(s![.., 0..num_features]).to_owned();
    let labels = original_data.column(num_features).to_owned();
    
    let linfa_dataset = Dataset::new(features, labels)
        .map_targets(|x| match x.to_owned() as i32 {
            i32::MIN..=4 => "Sad",
            5..=7 => "Ok",
            8..=i32::MAX => "Happy",
        })
        .with_feature_names(feature_names);
    
    let model = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .fit(&linfa_dataset)
        .unwrap();
    
    File::create("dt.tex")
        .unwrap()
        .write_all(model.export_to_tikz().with_legend().to_string().as_bytes())
        .unwrap();
}
