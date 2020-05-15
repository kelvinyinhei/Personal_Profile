import os
import datetime
import pandas as pd


class FuturesRolloverDataMerger(object):
    def __init__(self, symbol_rollover_plan: dict, export_file_name: str):
        self.export_folder_name = 'merged_data'
        self.symbol_rollover_plan = symbol_rollover_plan
        self.export_file_name = export_file_name
        # File existence checking
        for profile_dict in self.symbol_rollover_plan:
            file_name = profile_dict['file_name']
            if not os.path.isfile(file_name):
                raise FileNotFoundError('No such file: %s' % file_name)
        # build directory
        if not os.path.isdir(self.export_folder_name):
            os.mkdir(self.export_folder_name)

    @staticmethod
    def get_dataframe_given_time_range(file_name: str, time_start: datetime.datetime,
                                       time_end: datetime.datetime) -> pd.DataFrame:
        '''
        Read a file as pandas dataframe by file_name, export dataframe between time start and time end

        :param file_name: file path of data
        :param time_start: time start of the dataframe subset
        :param time_end: time end of the dataframe subset
        :return: pandas dataframe between time start and time end
        '''
        df = pd.read_csv(file_name)
        df['date'] = [datetime.datetime.strptime(dt, '%Y%m%d  %H:%M:%S') for dt in df['date']]
        df = df[df['date'] >= time_start]
        df = df[df['date'] <= time_end]
        df = df.sort_values('date')
        df['date'] = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in df['date']]
        return df

    def main(self) -> pd.DataFrame:
        '''
        Combine all the dataframe according to the rollover plan
        :return: combined dataframe
        '''
        df_list = list()
        for rollover_plan in self.symbol_rollover_plan:
            f_name = rollover_plan['file_name']
            print('Reading File: %s' % f_name)
            t_s = rollover_plan['start_date']
            t_e = rollover_plan['rollover_date']
            t_s = datetime.datetime.combine(t_s, datetime.time(9, 15, 0))
            t_e = datetime.datetime.combine(t_e, datetime.time(16, 30, 0))
            # extent the time to non regular trading hours
            # 16:30 to next day 03:00
            t_e += datetime.timedelta(hours=10, minutes=30)
            df = self.get_dataframe_given_time_range(f_name, t_s, t_e)
            df_list.append(df)
        all_df = pd.concat(df_list)
        colnames = ['Date Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        all_df.columns = colnames
        all_df = all_df.drop_duplicates('Date Time')
        # all_df.index = range(len(all_df))
        all_df['Adj Close'] = all_df['Close']
        file_path = os.path.join(self.export_folder_name, file_name)
        all_df.to_csv(file_path, index=False)
        print('File exported: %s' % file_path)
        return all_df


if __name__ == '__main__':
    symbol_rollover_plan = [
        {
            'file_name': 'data/MHI_MHIG17.csv',
            'start_date': datetime.datetime(2017, 1, 26),
            'rollover_date': datetime.datetime(2017, 2, 24)
        },
        {
            'file_name': 'data/MHI_MHIH17.csv',
            'start_date': datetime.datetime(2017, 2, 27),
            'rollover_date': datetime.datetime(2017, 3, 29)
        },
        {
            'file_name': 'data/MHI_MHIJ17.csv',
            'start_date': datetime.datetime(2017, 3, 30),
            'rollover_date': datetime.datetime(2017, 4, 26)
        },
        {
            'file_name': 'data/MHI_MHIK17.csv',
            'start_date': datetime.datetime(2017, 4, 27),
            'rollover_date': datetime.datetime(2017, 5, 26)
        },
        {
            'file_name': 'data/MHI_MHIM17.csv',
            'start_date': datetime.datetime(2017, 5, 29),
            'rollover_date': datetime.datetime(2017, 6, 28)
        },
        {
            'file_name': 'data/MHI_MHIN17.csv',
            'start_date': datetime.datetime(2017, 6, 29),
            'rollover_date': datetime.datetime(2017, 7, 27)
        },
        {
            'file_name': 'data/MHI_MHIQ7.csv',
            'start_date': datetime.datetime(2017, 7, 28),
            'rollover_date': datetime.datetime(2017, 8, 29)
        },
        {
            'file_name': 'data/MHI_MHIU7.csv',
            'start_date': datetime.datetime(2017, 8, 30),
            'rollover_date': datetime.datetime(2017, 9, 27)
        },
        {
            'file_name': 'data/MHI_MHIV7.csv',
            'start_date': datetime.datetime(2017, 9, 28),
            'rollover_date': datetime.datetime(2017, 10, 27)
        },
        {
            'file_name': 'data/MHI_MHIX7.csv',
            'start_date': datetime.datetime(2017, 10, 30),
            'rollover_date': datetime.datetime(2017, 11, 28)
        },
        {
            'file_name': 'data/MHI_MHIZ7.csv',
            'start_date': datetime.datetime(2017, 11, 29),
            'rollover_date': datetime.datetime(2017, 12, 27)
        },
        {
            'file_name': 'data/MHI_MHIF8.csv',
            'start_date': datetime.datetime(2017, 12, 28),
            'rollover_date': datetime.datetime(2018, 1, 29)
        },
        {
            'file_name': 'data/MHI_MHIG8.csv',
            'start_date': datetime.datetime(2018, 1, 30),
            'rollover_date': datetime.datetime(2018, 2, 26)
        },
        {
            'file_name': 'data/MHI_MHIH8.csv',
            'start_date': datetime.datetime(2018, 2, 27),
            'rollover_date': datetime.datetime(2018, 3, 27)
        },
        {
            'file_name': 'data/MHI_MHIJ8.csv',
            'start_date': datetime.datetime(2018, 3, 28),
            'rollover_date': datetime.datetime(2018, 4, 26)
        },
        {
            'file_name': 'data/MHI_MHIK8.csv',
            'start_date': datetime.datetime(2018, 4, 27),
            'rollover_date': datetime.datetime(2018, 5, 29)
        },
        {
            'file_name': 'data/MHI_MHIM8.csv',
            'start_date': datetime.datetime(2018, 5, 30),
            'rollover_date': datetime.datetime(2018, 6, 27)
        },
        {
            'file_name': 'data/MHI_MHIN8.csv',
            'start_date': datetime.datetime(2018, 6, 28),
            'rollover_date': datetime.datetime(2018, 7, 28)
        },
        {
            'file_name': 'data/MHI_MHIQ8.csv',
            'start_date': datetime.datetime(2018, 7, 30),
            'rollover_date': datetime.datetime(2018, 8, 29)
        },
        {
            'file_name': 'data/MHI_MHIU8.csv',
            'start_date': datetime.datetime(2018, 8, 30),
            'rollover_date': datetime.datetime(2018, 9, 26)
        },
        {
            'file_name': 'data/MHI_MHIV8.csv',
            'start_date': datetime.datetime(2018, 9, 27),
            'rollover_date': datetime.datetime(2018, 10, 29)
        },
        {
            'file_name': 'data/MHI_MHIX8.csv',
            'start_date': datetime.datetime(2018, 10, 30),
            'rollover_date': datetime.datetime(2018, 11, 28)
        },
        {
            'file_name': 'data/MHI_MHIZ8.csv',
            'start_date': datetime.datetime(2018, 11, 29),
            'rollover_date': datetime.datetime(2018, 12, 27)
        },
        {
            'file_name': 'data/MHI_MHIF9.csv',
            'start_date': datetime.datetime(2018, 12, 28),
            'rollover_date': datetime.datetime(2019, 1, 29)
        }
    ]
    file_name = 'MHI_merged.csv'
    ib_data_handler = FuturesRolloverDataMerger(symbol_rollover_plan, file_name)
    ib_data_handler.main()