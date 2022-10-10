
use std::fs;

fn main() {
    let files = fs::read_dir("./").unwrap();
    for f in files {
        let name = String::from(f.unwrap().path().to_str().unwrap());
        if name.ends_with(".png") && name.starts_with("./tmpframe") {
            let mut num_string = String::from(name.strip_prefix("./tmpframe").unwrap().strip_suffix(".png").unwrap());
            while num_string.len() < 4 {
                num_string.insert(0, '0');
            }
            num_string.insert_str(0, "frame");
            num_string.push_str(".png");
            println!("Renaming {} to {}", name, num_string);
            fs::rename(name.strip_prefix("./").unwrap(), num_string);

        } else {
            println!("Skipping: {}", name);;
        }

    }

}
