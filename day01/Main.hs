module Main (main) where

import Data.List (sort)
import Lib (distance, readFileFromArgs)

main :: IO ()
main = do
  contents <- readFileFromArgs
  let (xs, ys) = unzip $ readPairs contents
  part1 (xs, ys)
  part2 (xs, ys)

part1 :: ([Int], [Int]) -> IO ()
part1 (xs, ys) = do
  let (xs', ys') = (sort xs, sort ys)
  let totalDistance = sum $ zipWith distance xs' ys'
  putStrLn $ "Part 1: " ++ show totalDistance

part2 :: ([Int], [Int]) -> IO ()
part2 (xs, ys) = do
  let totalScore = sum $ map (\x -> x * occurrences x ys) xs
  putStrLn $ "Part 2: " ++ show totalScore

readPairs :: String -> [(Int, Int)]
readPairs = map readPair . lines

readPair :: String -> (Int, Int)
readPair line = case words line of
  [x, y] -> (read x, read y)
  _ -> error "Invalid input"

occurrences :: Eq a => a -> [a] -> Int
occurrences = (length .) . filter . (==)
