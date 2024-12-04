let parse_pair line =
  let parts = String.split_on_char ' ' line |> List.filter (fun s -> s <> "") in
  match parts with
  | [a; b] -> (int_of_string a, int_of_string b)
  | _ -> failwith "Invalid input"

let parse_input input = 
  input
  |> String.split_on_char '\n'
  |> List.filter (fun s -> s <> "")
  |> List.map parse_pair
  |> List.split

let distance a b = abs (a - b)

let part1 input = 
  let (xs, ys) = parse_input input in
  let xs = List.sort compare xs in
  let ys = List.sort compare ys in
  let total = List.fold_left2 (fun acc x y -> acc + distance x y) 0 xs ys in
  total

let score x ys = x * (List.fold_left (fun acc y -> if x = y then acc + 1 else acc) 0 ys)

let part2 input = 
  let (xs, ys) = parse_input input in
  let total = List.fold_left (fun acc x -> acc + score x ys) 0 xs in
  total

let run part input =
  match part with
  | 1 -> Printf.printf "Part 1: %d\n" (part1 input)
  | 2 -> Printf.printf "Part 2: %d\n" (part2 input)
  | _ -> Printf.printf "Invalid part: %d\n" part