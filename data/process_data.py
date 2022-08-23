# import libraries
import pandas as pd
import numpy as np
import re
import sys
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load messages from disaster_messages and categories from distaster_categories csv file
    INPUT:
        messages_filepath - file path from disaster_messages
        categories_filepath - file path from disaster_categories
    OUTPUT:
        merge_df - merged df from messages and categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    merge_df = df = messages.merge(categories, on='id')

    return merge_df


def clean_data(df):
    '''
    This function cleans the data including the following steps
    1. Split categories into separate columns
    2. Convert category values to 0 and 1
    3. Remove duplicates
    INPUT:
        df - merged data frame of messages and categories
    OUTPUT:
        clean_df - cleaned data frame following steps 1-3
    '''

    # create a dataframe of the individual category columns
    categories = df.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # extract a list of new column names for categories
    category_colnames = row.map(lambda x: str(x)[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to numeric
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:]
        categories[column] = categories[column].astype('int64')
    categories = categories.replace(2,1)

    # drop the original categories column from `df`
    df=df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()


def save_data(df, database_filename):
    '''
    save data to sql database
    INPUT:
        df - data frame to save
        database_filename - data base name
    ''' 
    
    # create engine using sqlalchemy
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages_disaster_response', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()