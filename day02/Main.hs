module Main (main) where

import Lib (distance, readFileFromArgs)

-- input file:
-- lines of "reports"
-- each report is a line of numbers separated by spaces
-- each number is a "level"
-- a report is "safe" if both are true:
--  - levels are either ALL INCREASING or ALL DECREASING
--  - any two adjacent levels differe by at least ONE and at most THREE

main :: IO ()
main = do
  contents <- readFileFromArgs
  let reports = readReports contents
  part1 reports

part1 :: [[Int]] -> IO ()
part1 reports = do
  let safeReports = filter isSafe reports
  putStrLn $ "Part 1: " ++ show (length safeReports)

readReport :: String -> [Int]
readReport = map read . words

readReports :: String -> [[Int]]
readReports = map readReport . lines

isSafe :: [Int] -> Bool
isSafe [] = True
isSafe [_] = True
isSafe (x : y : xs) =
  checkDistance x y && case x `compare` y of
    LT -> isSafeImpl (y : xs) LT
    EQ -> False
    GT -> isSafeImpl (y : xs) GT

isSafeImpl :: [Int] -> Ordering -> Bool
isSafeImpl [] _ = True
isSafeImpl [_] _ = True
isSafeImpl (x : y : xs) op = checkDistance x y && x `compare` y == op && isSafeImpl (y : xs) op

between :: Ord a => a -> a -> a -> Bool
between lo hi z = lo <= z && z <= hi

checkDistance :: Int -> Int -> Bool
checkDistance x y = between 1 3 (distance x y)