use gte_qwen::{EmbdQwenError, EmbdQwenModel};

fn main() -> Result<(), EmbdQwenError> {
    let local_path = "/home/robin/github/gte_Qwen1.5-7B-instruct";

    let mut model = EmbdQwenModel::new(local_path)?;

    // Encode the queries and the targets
    let queries = vec!["how much protein should a female eat", "summit define"];
    let targets = vec!["As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    ];

    let queries: Vec<_> = queries.iter().map(|x| model.prepare(&x, true)).collect();
    let targets: Vec<_> = targets.iter().map(|x| model.prepare(&x, false)).collect();
    let queries: Vec<&str> = queries.iter().map(|x| x.as_ref()).collect();
    let targets: Vec<&str> = targets.iter().map(|x| x.as_ref()).collect();

    let queries = model.embedding(queries)?;
    let targets = model.embedding(targets)?;

    println!("{:?}", queries.shape());
    println!("{:?}", targets.shape());

    let scores = model.score(&queries, &targets)?;

    println!("{:?}", scores);

    Ok(())
}
