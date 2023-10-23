use std::{fs::File, io::Write};

use linfa::{Dataset, traits::Fit};
use linfa::prelude::Predict;
use linfa_trees::{DecisionTree, SplitQuality};
use ndarray::{Array2, Axis, array, s};
use csv::{ReaderBuilder, WriterBuilder};
use ndarray_csv::{Array2Reader, Array2Writer};
use std::error::Error;

fn main() {
    // let original_data: Array2<f32> = array!(
    //     [60.,   1.,     12.,    13.,     10.],
    //     [30.,   2.,     3.,     8.,      2.],
    //     [15.,   5.,     1.,     9.,      3.],
    //     [5.,    1.,     6.,     9.,      4.],
    //     [22.,   8.,     6.,     10.,     4.],
    //     [24.,   2.,     6.,     16.,     9.],
    //     [50.,   4.,     12.,    13.,     8.],
    //     [40.,   6.,     3.,     11.,     7.],
    //     [35.,   5.,     1.,     7.,      1.],
    //     [52.,   7.,     4.,     8.,      2.],
    //     [21.,   8.,     4.,     16.,     9.],
    //     [31.,   3.,     3.,     15.,     9.]
    // );

    let file = File::open("BankChurners_200_Int.csv").unwrap();
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(&file);
    let array_read: Array2<f32> = reader.deserialize_array2((199, 16)).unwrap();

    // Read headers after Deserialize. Get error for expected: 16 Got: 4.
    let mut reader2 = ReaderBuilder::new().has_headers(true).from_reader(&file);
    let feature_names = (&mut reader2).headers().unwrap().iter().collect::<Vec<&str>>();
    dbg!(&feature_names);

    dbg!(&array_read);

    // let feature_names = vec!["ConsultLength", "Consultant", "ConsultLocation", "HourOfDay", "ConsultRating"];
    let num_features = array_read.len_of(Axis(1)) - 1;
    let features = array_read.slice(s![.., 0..num_features]).to_owned();
    let labels = array_read.column(num_features).to_owned();
    
    let linfa_dataset = Dataset::new(features, labels)
        .map_targets(|x| match x.to_owned() as i32 {
            i32::MIN..=17 => "Poor",
            18..=35 => "Moderate",
            36..=i32::MAX => "Compelling",
        })
        .with_feature_names(feature_names);
    
    let model = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .fit(&linfa_dataset)
        .unwrap();

    let test: Array2<f32> = array!(
        [51., 1., 3., 5., 3., 4., 1., 36., 6., 2., 3., 11106., 2107., 8999., 1243., 24.]

    );
    let predictions = model.predict(&test);
    
    println!("{:?}", predictions);
    // println!("{:?}", test.targets);
    
    // File::create("dt.tex")
    //     .unwrap()
    //     .write_all(model.export_to_tikz().with_legend().to_string().as_bytes())
    //     .unwrap();
}
